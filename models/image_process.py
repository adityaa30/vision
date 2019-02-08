import numpy as np
import bcolz
import image_captioning.coco as coco
from image_captioning.utils import *
from image_captioning.config import Config

import keras
from tensorflow.python.keras import backend as K


class ImageProcess:
    def __init__(self, model, transfer_layer, batch_size, name, config):
        """
        ImageProcess is a helper class to process images under the hood
        to calculate the transfer-values for any model provided using Keras
        and TensorFlow APIs to support parallel-processing the images in a
        batch

        :param model: Model to calculate the transfer-values
        :param transfer_layer: string representing the output layer of transfer-model
        :param batch_size: Batch size for processing the images in the transfer-model
        :param config: Instance of Config class
        """
        assert isinstance(model, keras.models.Model)
        assert isinstance(config, Config)
        self.model = model
        self.batch_size = batch_size
        self.model_name = name
        self.config = config

        try:
            self.transfer_layer = model.get_layer(transfer_layer)
            self.transfer_values_size = K.int_shape(self.transfer_layer.output)[1]
            self.image_size = K.int_shape(self.model.inputs[0])[1:]
            self.transfer_model = keras.models.Model(
                inputs=self.model.inputs,
                outputs=self.transfer_layer.output
            )
        except ValueError as e:
            print(e)

    def _process_images(self, data_dir, filenames, bcolz_dir):
        """
        Process all the given files in the given data_dir using the
        pre-trained image-model and return their transfer-values.

        Note that we process the images in batches to save
        memory and improve efficiency on the GPU.
        """

        num_images = len(filenames)

        # Pre-allocate input-batch-array for images
        shape = (self.batch_size,) + self.image_size
        image_batch = np.zeros(shape=shape, dtype=np.float64)

        # Pre-allocate output-array for transfer-values
        shape = (num_images, self.transfer_values_size)
        transfer_values = bcolz.carray(np.zeros(shape=shape, dtype=np.float64), rootdir=bcolz_dir, mode='w')

        start_index = 0

        # Process batches of image-files
        while start_index < num_images:
            print_progress(count=start_index, max_count=num_images)

            end_index = start_index + self.batch_size
            if end_index > num_images:
                end_index = num_images
            current_batch_size = end_index - start_index

            # Load all the images in the batch
            for i, filename in enumerate(filenames[start_index:end_index]):
                path = os.path.join(data_dir, filename)
                img = coco.load_image(path, size=self.image_size[:2])
                image_batch[i] = img

            # Use the pre-trained image-model to process the image
            transfer_values_batch = self.transfer_model.predict(image_batch[0:current_batch_size])

            # Save the transfer-values
            transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]

            start_index = end_index

        print()

        return transfer_values

    def process_images(self, filenames, train=True):
        """
        Processes all the images in the training/cross-validation set and
        saves these transfer-values in a cache-file for faster reloading

        :param filenames: list of all the training/val images filenames
        :param train: True if @filenames refer to training data
                      False if @filenames refer to cross-validation data
        :return: Transfer values
        """
        print("\nProcessing {0} images in training-set ...".format(len(filenames)))

        if train:
            cache_path = os.path.join(self.config.paths.DATASET_DIR, f'{self.model_name}_transfer_values_train.pkl')
            data_dir = self.config.paths.TRAIN_DIR
            bcolz_dir = os.path.join(
                self.config.paths.DATASET_DIR,
                f'{self.model_name}_{self.config.paths.BCOLZ_TRAIN_TRANSFER_VALUES}'
            )
        else:
            cache_path = os.path.join(self.config.paths.DATASET_DIR, f'{self.model_name}_transfer_values_val.pkl')
            data_dir = self.config.paths.VAL_DIR
            bcolz_dir = os.path.join(
                self.config.paths.DATASET_DIR,
                f'{self.model_name}_{self.config.paths.BCOLZ_VAL_TRANSFER_VALUES}'
            )

        # If the cache-file already exists then reload it,
        # otherwise process all images and save their transfer-values
        # to the cache-file so it can be reloaded quickly.
        transfer_values = cache(
            cache_path=cache_path,
            fn=self._process_images,
            data_dir=data_dir,
            filenames=filenames,
            bcolz_dir=bcolz_dir
        )

        return transfer_values


if __name__ == '__main__':
    # Process the pre-trained VGG16 model
    vgg16_model = keras.applications.vgg16.VGG16(
        weights='imagenet'
    )
    assert isinstance(vgg16_model, keras.models.Model)

    vgg16_model.summary()
    config = Config()

    train_ids, train_filenames, train_captions = coco.load_records(train=True)
    val_ids, val_filenames, val_captions = coco.load_records(train=False)
    image_process = ImageProcess(vgg16_model, 'fc2', 16, 'vgg16', config)
