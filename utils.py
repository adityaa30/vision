from config import Config

import pickle
import os
import sys
import numpy as np
import bcolz


def print_list(list):
    """
    Displays each element of the list in separate line
    :param list: 1D list containing elements
    """
    output_string = ''
    for item in list:
        output_string += str(item) + '\n'

    return output_string


def mark_captions(captions_list, mark_start='<SOS> ', mark_end=' <EOS>'):
    """
    Wraps all text-strings in the start[*]end markers.

    :param captions_list: list of lists with text-captions
    :param mark_start: starting marker for each caption
    :param mark_end: ending marker for each caption
    """
    captions_marked = [
        [
            mark_start + caption + mark_end
            for caption in captions
        ]
        for captions in captions_list
    ]

    return captions_marked


def flatten_captions(captions_list):
    """
    Converts a list-of-list to a flattened list of captions.

    :param captions_list: list of lists with text-captions
    """
    captions_list = [
        caption
        for captions in captions_list
        for caption in captions
    ]

    return captions_list


def print_progress(count, max_count):
    percentage_complete = count / max_count

    msg = f'\n-Processed : {count}/{max_count}\tProgress: {percentage_complete}'

    sys.stdout.write(msg)
    sys.stdout.flush()


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


def create_dataset(transfer_values, captions, config, train=True):
    """
    Creates a list of length = total number of captions where every list item
    is a list of size 2 -> [corresponding filepath path, caption]

    :param transfer_values: List of transfer-values for images
    :param captions: List of list of captions (tokenize) for each filename
    :param config: Instance of Config class
    :param train:
            True if transfer-values and captions are of training dataset
            False if transfer-values and captions are of cross-validation dataset
    """
    assert len(captions) == len(transfer_values)
    assert isinstance(config, Config)

    num_captions = len(captions)
    num_captions_list = [len(caption) for caption in captions]
    total_captions = np.sum(num_captions_list)

    dataset = np.zeros((total_captions, 2))
    if train:
        dataset = bcolz.carray(dataset, rootdir=config.paths.BCOLZ_TRAIN_DATASET, mode='w')
    else:
        dataset = bcolz.carray(dataset, rootdir=config.paths.BCOLZ_VAL_DATASET, mode='w')

    row_idx = 0  # Temporary variable to keep track of each row in the dataset
    for i in range(num_captions):
        for x in range(num_captions_list[i]):
            dataset[row_idx] = [i, x]
            row_idx += 1

    if train:
        print(f'Train Dataset created of shape : {dataset.shape}')
    else:
        print(f'Cross-validation Dataset created of shape : {dataset.shape}')

    return dataset
