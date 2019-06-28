import os


class Config:
    def __init__(self):
        self.paths = self.PATHS()

        # Batch size for training the COCO dataset
        # Same for both train and cross-validation data
        self.TRAIN_BATCH_SIZE = 32
        self.PADDING_FILL_VALUE = 0

        self.MARKER_START = 'ssss '
        self.MARKER_END = ' eeee'

    class PATHS:
        DATASET_DIR = 'dataset'

        TRAIN_DIR = os.path.join(DATASET_DIR, 'train2017')
        VAL_DIR = os.path.join(DATASET_DIR, 'val2017')
        ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annotations')

        TRAIN_DATASET = os.path.join(
            DATASET_DIR, 'train_transferred_dataset.pkl')
        VAL_DATASET = os.path.join(DATASET_DIR, 'val_transferred_dataset.pkl')

        BCOLZ_DIR = os.path.join(DATASET_DIR, 'bcolz')
        BCOLZ_TRAIN_CAPTIONS = os.path.join(BCOLZ_DIR, 'train_captions')
        BCOLZ_TRAIN_TRANSFER_VALUES = 'bcolz_transfer_values_train'

        BCOLZ_VAL_CAPTIONS = os.path.join(BCOLZ_DIR, 'val_captions')
        BCOLZ_VAL_TRANSFER_VALUES = 'bcolz_transfer_values_val'

        BCOLZ_TRAIN_DATASET = os.path.join(BCOLZ_DIR, 'train_dataset')
        BCOLZ_VAL_DATASET = BCOLZ_DIR + 'val_dataset'
