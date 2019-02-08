class Config:
    def __init__(self):
        self.paths = self.PATHS()

        # Batch size for training the COCO dataset
        # Same for both train and cross-validation data
        self.TRAIN_BATCH_SIZE = 32
        self.PADDING_FILL_VALUE = 0

    class PATHS:
        def __init__(self):
            self.DATASET_DIR = 'dataset/'

            self.TRAIN_DIR = self.DATASET_DIR + 'train/2017'
            self.VAL_DIR = self.DATASET_DIR + 'val/2017'

            self.BCOLZ_DIR = self.DATASET_DIR + 'bcolz/'
            self.BCOLZ_TRAIN_CAPTIONS = self.BCOLZ_DIR + 'train_captions'
            self.BCOLZ_TRAIN_TRANSFER_VALUES = 'bcolz_transfer_values_train'

            self.BCOLZ_VAL_CAPTIONS = self.BCOLZ_DIR + 'val_captions'
            self.BCOLZ_VAL_TRANSFER_VALUES = 'bcolz_transfer_values_val'

            self.BCOLZ_TRAIN_DATASET = self.BCOLZ_DIR + 'train_dataset'
            self.BCOLZ_VAL_DATASET = self.BCOLZ_DIR + 'val_dataset'
