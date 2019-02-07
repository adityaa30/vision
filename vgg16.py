import sys
import os
import numpy as np
import image_captioning.coco as coco


def print_progress(count, max_count):
    percentage_complete = count / max_count

    # Status-message
    # \r which means the line should overwrite itself
    msg = "\r- Progress: {0:.1%}".format(percentage_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()


def _process_images(data_dir, filenames, transfer_model, batch_size=16, img_size=(224, 224), transfer_values_size=4096):
    """
    Process all the given files in the given data_dir using the
    pre-trained image-model and return their transfer-values.

    Note that we process the images in batches to save
    memory and improve efficiency on the GPU.
    """

    num_images = len(filenames)

    # Pre-allocate input-batch-array for images
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float64)

    # Pre-allocate output-array for transfer-values
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float64)

    start_index = 0

    # Process batches of image-files
    while start_index < num_images:
        print_progress(count=start_index, max_count=num_images)

        end_index = start_index + batch_size
        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index

        # Load all the images in the batch
        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = coco.load_image(path, size=img_size)
            image_batch[i] = img

        # Use the pre-trained image-model to process the image
        transfer_values_batch = transfer_model.predict(image_batch[0:current_batch_size])

        # Save the transfer-values
        transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]

        start_index = end_index

    print()

    return transfer_values


def process_images(transfer_model, filenames, train=True):
    """
    Processes all the images in the training/cross-validation set and
    saves these transfer-values in a cache-file for faster reloading

    :param transfer_model: Model to calculate the transfer-values
    :param filenames: list of all the training/val images filenames
    :param train: True if @filenames refer to training data
                  False if @filenames refer to cross-validation data
    :return: Transfer values
    """
    print("\nProcessing {0} images in training-set ...".format(len(filenames)))

    if train:
        cache_path = os.path.join(coco.DOWNLOAD_DIR, "vgg16_transfer_values_train.pkl")
        data_dir = coco.TRAIN_DIR
    else:
        cache_path = os.path.join(coco.DOWNLOAD_DIR, "vgg16_transfer_values_val.pkl")
        data_dir = coco.VAL_DIR

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = coco.cache(cache_path=cache_path,
                                 fn=_process_images,
                                 data_dir=data_dir,
                                 filenames=filenames,
                                 transfer_model=transfer_model)

    return transfer_values
