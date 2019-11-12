import os
import re
from random import randrange
import numpy as np
import tensorflow as tf

from vision.models.tokenizer import TokenizerWrapper

KA = tf.keras.applications
KP = tf.keras.preprocessing


PATH_DATASET = 'flickr'
PATH_IMAGES = os.path.join(PATH_DATASET, 'images')
PATH_ENCODED_IMAGES = os.path.join(PATH_DATASET, 'encoded_images')
PATH_TOKENS = os.path.join(PATH_DATASET, 'tokens.txt')

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_records():
    regex = r'^(.*?)#.\t(.*?)\n$'
    filepaths, caption = [], []
    with open(PATH_TOKENS, 'r') as f:
        data = f.readlines()
        for dat in data:
            temp = re.findall(regex, dat)[0]
            filepaths.append(os.path.join(PATH_ENCODED_IMAGES, temp[0]) + '.npy')
            caption.append(f'<start> {temp[1]} <end>')
    
    return filepaths, caption


def load_image(path):
    """
    Load the image from the given file-path and resize it
    to the size compatible with MobileNetV2
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, path


class FlickrDataset:
    def __init__(self, batch_size=48, encode_inp=False):
        self.batch_size = batch_size

        # Create extract model
        model = KA.MobileNetV2(weights='imagenet', include_top=False)
        self.extract_model = tf.keras.Model(
            model.input, model.layers[-1].output)


        if encode_inp:
            self.save_encoded_images()

        self.filepaths, self.captions = load_records()
        self.tokenizer = TokenizerWrapper(self.captions)
        caps_token = self.tokenizer.texts_to_sequences(self.captions)
        caps_token = KP.sequence.pad_sequences(caps_token)

        self.max_len = max([len(cap) for cap in caps_token])

        self.train_dataset = tf.data.Dataset \
            .from_tensor_slices((self.filepaths, caps_token)) \
            .map(lambda item1, item2: tf.numpy_function(self.map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=AUTOTUNE) \
            .shuffle(1000, reshuffle_each_iteration=True) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(buffer_size=AUTOTUNE)

    @staticmethod
    def map_func(img_path, cap):
        # Load the numpy files
        # img_name & cap is of type: tf.string
        img_tensor = np.load(img_path)
        return img_tensor, cap

    def get_random_path(self):
        idx = randrange(len(self.filepaths))
        path = os.path.basename(self.filepaths[idx])[:-4]
        path = os.path.join(PATH_IMAGES, path)
        caption = []
        for i in range(len(self.filepaths)):
            if self.filepaths[idx] == self.filepaths[i]:
                caption.append(self.captions[i].split(' ')[1:-2])

        return path, caption
        

    def save_encoded_images(self):
        set_filepaths = []
        for path in os.listdir(PATH_IMAGES):
            set_filepaths.append(os.path.join(PATH_IMAGES, path))
            
        dataset = tf.data.Dataset \
            .from_tensor_slices(list(set_filepaths)) \
            .map(load_image, num_parallel_calls=AUTOTUNE) \
            .batch(self.batch_size)
        
        count = 0
        print(f'Processed {count} images out of {len(set_filepaths)}')
        
        for img, path in dataset:
            batch_features = self.extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                count += 1
                path_of_feature = p.numpy().decode('utf-8')
                path_of_feature = os.path.join(
                    PATH_ENCODED_IMAGES, 
                    os.path.basename(path_of_feature))
                np.save(path_of_feature, bf.numpy())
            
                if count % 400 == 0:
                    print(f'Processed {count} images out of {len(set_filepaths)}')
