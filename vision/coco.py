from extract import extract
from utils import *
from config import Config

import os
import json
import tensorflow as tf
import numpy as np

config = Config()
DOWNLOAD_DIR = 'dataset/'

TRAIN_DIR = DOWNLOAD_DIR + 'train2017/'
VAL_DIR = DOWNLOAD_DIR + 'val2017/'

FILES = [
    ['train2017.zip', config.paths.TRAIN_DIR],
    ['val2017.zip', config.paths.VAL_DIR],
    ['annotations_trainval2017.zip', config.paths.ANNOTATION_DIR]
]


# Extract the dataset

def extract_files():
    for zip_file, file in FILES:
        if os.path.exists(file):
            print('{} already unpacked'.format(zip_file))
        else:
            extract(zip_file, DOWNLOAD_DIR)


def _load_records(train=True):
    """
    Load the image-filenames and captions
    for either the training set or the cross-validation set.
    """

    if train:
        filename = "captions_train2017.json"
    else:
        filename = "captions_val2017.json"

    # Load the data-file
    path = os.path.join(config.paths.DATASET_DIR, "annotations", filename)
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    images = data_raw['images']
    annotations = data_raw['annotations']

    # dict for holding our data.
    # lookup-key is the image-id.
    records = dict()

    for image in images:
        image_id = image['id']
        filename = image['file_name']

        record = dict()
        record['filename'] = filename
        record['captions'] = list()
        records[image_id] = record  # image id as the key

    for ann in annotations:
        image_id = ann['image_id']
        caption = '<start>' + ann['caption'] + '<end>'

        record = records[image_id]
        record['captions'].append(caption)

    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    # Convert the list of tuples to separate tuples with the data.
    ids, filenames, captions = zip(*records_list)

    return ids, filenames, captions


def load_records(train=True):
    """
    :param train: When True loads the training data, else the cross-validation data
    """
    if train:
        cache_filename = "records_train.pkl"
    else:
        cache_filename = "records_val.pkl"

    cache_path = os.path.join(DOWNLOAD_DIR, cache_filename)
    records = cache(cache_path=cache_path,
                    fn=_load_records,
                    train=train)

    return records


class COCODataset:
    def __init__(self, batch_size=16, train_model='mobilenetv2'):
        train_ids, train_filenames, train_captions = load_records(train=True)
        val_ids, val_filenames, val_captions = load_records(train=False)

        if train_model == 'mobilenetv2':
            preprocess_fn = self.load_image_mobilenet

        self.train_image_dataset = tf.data.Dataset.from_tensor_slices(train_filenames). \
            map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size)

        self.val_image_dataset = tf.data.Dataset.from_tensor_slices(val_filenames). \
            map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size)

    @staticmethod
    def load_image_mobilenet(path):
        """
        Load the image from the given file-path and resize it
        to the size compatible with MobileNetV2
        """

        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, path


    def dataset(self, train=True):
        if train:
            return self.train_image_dataset
        else:
            return self.val_image_dataset