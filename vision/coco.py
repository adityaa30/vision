import os
import json
import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle

from vision.extract import extract
from vision.utils.utils import cache


class PATHS:
    DATASET_DIR = 'dataset'

    TRAIN_DIR = os.path.join(DATASET_DIR, 'train2017')
    VAL_DIR = os.path.join(DATASET_DIR, 'val2017')
    ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annotations')

    CACHE_DIR = os.path.join(DATASET_DIR, 'cache')
    CACHE_TRAIN_RECORDS = os.path.join(CACHE_DIR, 'records_train.pkl')
    CACHE_VAL_RECORDS = os.path.join(CACHE_DIR, 'records_val.pkl')
    CACHE_TRAIN_TRANSFER = os.path.join(
        CACHE_DIR, 'train_transferred_dataset.pkl')
    CACHE_VAL_TRANSFER = os.path.join(CACHE_DIR, 'val_transferred_dataset.pkl')


FILES = [
    ['train2017.zip', PATHS.TRAIN_DIR],
    ['val2017.zip', PATHS.VAL_DIR],
    ['annotations_trainval2017.zip', PATHS.ANNOTATION_DIR]
]


# Extract the dataset
def extract_files(path):
    for zip_file, file in FILES:
        if os.path.exists(file):
            print('{} already unpacked'.format(zip_file))
        else:
            extract(zip_file, path)


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
    path = os.path.join(PATHS.ANNOTATION_DIR, filename)
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
        caption = '<start> ' + ann['caption'] + ' <end>'

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
        cache_path = PATHS.CACHE_TRAIN_RECORDS
    else:
        cache_path = PATHS.CACHE_VAL_RECORDS

    records = cache(cache_path=cache_path,
                    fn=_load_records,
                    train=train)

    return records


class COCODataset:
    def __init__(self, batch_size=16, train_model='mobilenetv2'):
        _, self.train_filenames, self.train_captions = load_records()
        _, self.val_filenames, self.val_captions = load_records(False)

        if train_model == 'mobilenetv2':
            load_fn = self.load_image_mobilenet

        self.train_image_dataset = tf.data.Dataset.from_tensor_slices(
            list(self.train_filenames)
        )
        self.train_image_dataset = self.train_image_dataset.map(
            self.load_image_mobilenet,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

        self.val_image_dataset = tf.data.Dataset.from_tensor_slices(
            list(self.val_filenames)
        )
        self.val_image_dataset = self.val_image_dataset.map(
            self.load_image_mobilenet,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

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
