from models.image_process import ImageProcess
from utils import create_dataset

import keras
import os
import bcolz
import pickle


class VGG16:
    def __init__(
            self,
            batch_size,
            train_filenames,
            val_filenames,
            train_tokens,
            val_tokens,
            config,
    ):
        self.batch_size = batch_size
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.config = config

        self.model = keras.applications.vgg16.VGG16(weights='imagenet')
        self.model.summary()

        self.image_process = ImageProcess(
            model=self.model,
            transfer_layer='fc2',
            batch_size=self.batch_size,
            name='vgg16',
            config=self.config
        )

        self.transfer_values_train = None
        self.load_train_transfer_values()

        self.transfer_values_val = None
        self.load_val_transfer_values()

        # Create the dataset we'll be using to train
        # Dataset is a list with each item as a list of
        # type : [transfer-value, caption (tokenize)]
        if os.path.exists(config.paths.TRAIN_DATASET):

            with open(config.paths.TRAIN_DATASET, mode='rb') as file:
                self.train_dataset = pickle.load(file)

            print(
                f'Loaded cached train dataset from {config.paths.TRAIN_DATASET} of shape : {self.train_dataset.shape}')
        else:
            print('Creating train dataset : ')
            train_dataset = create_dataset(
                transfer_values=self.transfer_values_train,
                captions=self.train_tokens,
                config=config
            )
            print(f'Shape of Training dataset : {train_dataset.shape}')

            self.train_dataset = bcolz.carray(train_dataset, rootdir=config.paths.BCOLZ_TRAIN_DATASET, mode='w')

            with open(config.paths.TRAIN_DATASET, mode='wb') as file:
                pickle.dump(train_dataset, file)

            print(f'Save Train dataset to {config.paths.TRAIN_DATASET}')

        if os.path.exists(config.paths.VAL_DATASET):

            with open(config.paths.VAL_DATASET, mode='rb') as file:
                self.val_dataset = pickle.load(file)

            print(
                f'Loaded cached train dataset from {config.paths.VAL_DATASET} of shape : {self.val_dataset.shape}')
        else:

            print('Creating cross-validation dataset : ')
            val_dataset = create_dataset(
                transfer_values=self.transfer_values_val,
                captions=self.val_tokens,
                config=config
            )
            print(f'Shape of Cross-validation dataset : {val_dataset.shape}')

            self.val_dataset = bcolz.carray(val_dataset, rootdir=config.paths.BCOLZ_VAL_DATASET, mode='w')

            with open(config.paths.VAL_DATASET, mode='wb') as file:
                pickle.dump(self.val_dataset, file)

            print(f'Save Train dataset to {config.paths.VAL_DATASET}')

    def load_train_transfer_values(self):
        self.transfer_values_train = self.image_process.process_images(self.train_filenames, train=True)
        print('\nTransfer values train : ')
        print("dtype:", self.transfer_values_train.dtype)
        print("shape:", self.transfer_values_train.shape)

    def load_val_transfer_values(self):
        self.transfer_values_val = self.image_process.process_images(self.val_filenames, train=False)
        print('\nTransfer values cross-validation : ')
        print("dtype:", self.transfer_values_val.dtype)
        print("shape:", self.transfer_values_val.shape)


if __name__ == '__main__':
    pass
