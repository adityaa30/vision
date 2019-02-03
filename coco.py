from image_captioning.extract import extract
import os
import pickle
import json
from PIL import Image
import numpy as np

DOWNLOAD_DIR = 'dataset/'

TRAIN_DIR = DOWNLOAD_DIR + 'train2017/'
VAL_DIR = DOWNLOAD_DIR + 'val2017/'

FILES = [
    ['train2017.zip', DOWNLOAD_DIR + 'train2017/'],
    ['val2017.zip', DOWNLOAD_DIR + 'val2017/'],
    ['annotations_trainval2017.zip', DOWNLOAD_DIR + 'annotations/']
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
    path = os.path.join(DOWNLOAD_DIR, "annotations", filename)
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
        caption = ann['caption']

        record = records[image_id]
        record['captions'].append(caption)

    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    # Convert the list of tuples to separate tuples with the data.
    ids, filenames, captions = zip(*records_list)

    return ids, filenames, captions


def cache(cache_path, fn, *args, **kwargs):
    """
    Cache-wrapper for a function or class. If the cache-file exists
    then the data is reloaded and returned, otherwise the function
    is called and the result is saved to cache. The fn-argument can
    also be a class instead, in which case an object-instance is
    created and saved to the cache-file.

    :param cache_path: File-path for the cache-file.
    :param fn: Function or class to be called.
    :param args: Arguments to the function or class-init.
    :param kwargs: Keyword arguments to the function or class-init.
    :return: The result of calling the function or creating the object-instance.
    """

    if os.path.exists(cache_path):
        # Cache-file exist
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)
        print("- Data loaded from cache-file: " + cache_path)
    else:
        # Cache-file does not exist

        obj = fn(*args, **kwargs)

        # Save data to a cache-file
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


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


def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """

    img = Image.open(path)

    if size is not None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)

    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img
