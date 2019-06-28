import os
import pickle
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from vision.coco import COCODataset
from vision.utils.logger import Logger

class MobileNetV2:

    SAVE_PATH = os.path.join('data', 'transfer_vals')
    TRAIN_PATH = os.path.join(SAVE_PATH, 'train')
    VAL_PATH = os.path.join(SAVE_PATH, 'val')

    def __init__(self):
        self._model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, weights='imagenet')

        self.transfer_model = tf.keras.Model(
            self._model.input,
            self._model.layers[-1].output
        )

        self.dataset = COCODataset(train_model='mobilenetv2')

        self.LOGGER = Logger('MobileNetV2')

    def save_path(self, train=True):
        return self.TRAIN_PATH if train else self.VAL_PATH

    def decode_tranfer_vals(self, train=True):
        self.LOGGER.v('Starting to decode transfer values...')

        for img, path in tqdm(self.dataset.dataset(train)):
            batch_features = self.transfer_model(img)
            batch_features = tf.reshape(
                batch_features,
                (batch_features.shape[0], -1, batch_features.shape[3])
            )

            for bf, p in zip(batch_features, path):
                path_of_feature = os.path.join(
                    self.save_path(train),
                    p.numpy().decode('utf-8')
                )
                np.save(path_of_feature, bf.numpy())

        self.LOGGER.v('Finished decoding transfer values')