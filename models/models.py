import keras
from image_captioning.models.image_process import ImageProcess


class VGG16:
    def __init__(self, batch_size, train_filenames, val_filenames):
        self.batch_size = batch_size
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        self.model = keras.applications.vgg16.VGG16(weights='imagenet')
        self.model.summary()

        self.image_process = ImageProcess(
            model=self.model,
            transfer_layer='fc2',
            batch_size=self.batch_size,
            name='vgg16'
        )

        self.transfer_values_train = self.image_process.process_images(train_filenames, train=True)
        print('\nTransfer values train : ')
        print("dtype:", self.transfer_values_train.dtype)
        print("shape:", self.transfer_values_train.shape)

        self.transfer_values_val = self.image_process.process_images(val_filenames, train=False)
        print('\nTransfer values cross-validation : ')
        print("dtype:", self.transfer_values_val.dtype)
        print("shape:", self.transfer_values_val.shape)


class InceptionResNetV2:
    def __init__(self, batch_size, train_filenames, val_filenames):
        self.batch_size = batch_size
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        self.model = keras.applications.vgg16.VGG16(weights='imagenet')
        self.model.summary()

        self.image_process = ImageProcess(
            model=self.model,
            transfer_layer='fc2',
            batch_size=self.batch_size,
            name='inception_resent_v2'
        )

        self.transfer_values_train = self.image_process.process_images(train_filenames, train=True)
        print('\nTransfer values train : ')
        print("dtype:", self.transfer_values_train.dtype)
        print("shape:", self.transfer_values_train.shape)

        self.transfer_values_val = self.image_process.process_images(val_filenames, train=False)
        print('\nTransfer values cross-validation : ')
        print("dtype:", self.transfer_values_val.dtype)
        print("shape:", self.transfer_values_val.shape)
