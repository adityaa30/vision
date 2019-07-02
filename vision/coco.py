import os
import random
import json
import tensorflow as tf
import numpy as np

from vision.extract import extract
from vision.models.tokenizer import TokenizerWrapper
from vision.utils.utils import cache


class PATHS:
    DATASET_DIR = 'dataset'

    TRAIN_DIR = os.path.join(DATASET_DIR, 'train2017')
    VAL_DIR = os.path.join(DATASET_DIR, 'val2017')
    ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annotations')

    CACHE_DIR = os.path.join(DATASET_DIR, 'cache')
    CACHE_TRAIN_RECORDS = os.path.join(CACHE_DIR, 'records_train.pkl')
    CACHE_VAL_RECORDS = os.path.join(CACHE_DIR, 'records_val.pkl')

    TRANSFER_DIR = os.path.join(DATASET_DIR, 'transfer_vals')
    TRAIN_TRANSFER_DIR = os.path.join(TRANSFER_DIR, 'train')
    VAL_TRANSFER_DIR = os.path.join(TRANSFER_DIR, 'val')


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

    parent_dir = PATHS.TRAIN_DIR if train else PATHS.VAL_DIR
    for image in images:
        image_id = image['id']
        filename = os.path.join(parent_dir, )

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
    def __init__(self, batch_size=48, train_model='mobilenetv2'):
        _, self.train_filenames, self.train_captions = load_records()
        _, self.val_filenames, self.val_captions = load_records(False)
        del _

        self.batch_size = batch_size

        if train_model == 'mobilenetv2':
            load_fn = self.load_image_mobilenet

        self.transfer_train_dataset = tf.data.Dataset.from_tensor_slices(list(self.train_filenames))
        self.transfer_train_dataset = self.transfer_train_dataset.map(
            load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

        self.transfer_val_dataset = tf.data.Dataset.from_tensor_slices(list(self.val_filenames))
        self.transfer_val_dataset = self.transfer_val_dataset.map(
            load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

        # Create a Tokenizer
        self.tokenizer = TokenizerWrapper(self.get_texts())

        # Create a list of filenames each mapped with its corresponding caption
        _train_filenames, _train_captions = [], []
        for i, captions in enumerate(self.train_captions, start=0):
            # get the path of train transfer features
            train_path = os.path.join(PATHS.TRAIN_TRANSFER_DIR, os.path.basename(self.train_filenames[i])) + '.npy'
            for cap in captions:
                _train_filenames.append(train_path)
                _train_captions.append(cap)
        _train_captions = self.tokenizer.texts_to_sequences(_train_captions)
        _train_captions = tf.keras.preprocessing.sequence.pad_sequences(_train_captions, padding='post')
        max_len = max([len(cap) for cap in _train_captions])
        self.train_dataset = tf.data.Dataset.from_tensor_slices((_train_filenames, _train_captions)).map(
            lambda item1, item2: tf.numpy_function(self.map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(1000, reshuffle_each_iteration=True) \
            .batch(self.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # Free the memory
        del _train_filenames
        del _train_captions

        _val_filenames, _val_captions = [], []
        for i, captions in enumerate(self.val_captions, start=0):
            # get the path of train transfer features
            val_path = os.path.join(PATHS.VAL_TRANSFER_DIR, os.path.basename(self.val_filenames[i])) + '.npy'
            for cap in captions:
                _val_filenames.append(val_path)
                _val_captions.append(cap)
        _val_captions = self.tokenizer.texts_to_sequences(_val_captions)
        _val_captions = tf.keras.preprocessing.sequence.pad_sequences(_val_captions, padding='post')
        max_len = max([len(cap) for cap in _val_captions])
        self.val_dataset = tf.data.Dataset.from_tensor_slices((_val_filenames, _val_captions)) \
            .map(lambda item1, item2: tf.numpy_function(self.map_func, [item1, item2], [tf.float32, tf.int32]),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(1000, reshuffle_each_iteration=True) \
            .batch(self.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # Free the memory
        del _val_filenames
        del _val_captions

    @staticmethod
    def map_func(img_path, cap):
        # Load the numpy files
        # img_name & cap is of type: tf.string
        img_tensor = np.load(img_path)
        return img_tensor, cap

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

    def get_dataset(self, train=True):
        if train:
            return self.transfer_train_dataset
        else:
            return self.transfer_val_dataset

    def get_texts(self):
        texts = []
        _captions = self.train_captions + self.val_captions
        for caps in _captions:
            for cap in caps:
                texts.append(cap)

        return texts

    @staticmethod
    def caption_to_target(caption):
        # TODO: Convert caption -> target
        print(caption)
