import image_captioning.coco as coco
from image_captioning.models.models import VGG16, InceptionV3
from image_captioning.tokenizer import *
from image_captioning.glove import GloVe
from image_captioning.utils import *
from image_captioning.models.train import TrainModel, COCOSequenceGenerator
from image_captioning.config import Config

import random
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
STATE_SIZE = 512
BATCH_SIZE = 128
PATH_TRAINED_WEIGHTS = 'trained_weights/model-1.ckpt'
PATH_TENSORBOARD_LOGS = 'tensorboard/logs-1/'

# Create instance for config file
config = Config()

# Extract the files
coco.extract_files()

train_ids, train_filenames, train_captions = coco.load_records(train=True)
val_ids, val_filenames, val_captions = coco.load_records(train=False)

num_train_images = len(train_filenames)

print('Total number of training images : {}'.format(len(train_filenames)))
print('Total number of cross-validation images : {}'.format(len(val_filenames)))


def show_image(idx, train):
    """
    Load and plot an image from the training- or validation-set
    with the given index.
    """

    if train:
        # Use an image from the training-set.
        dir = coco.TRAIN_DIR
        filename = train_filenames[idx]
        captions = train_captions[idx]
    else:
        # Use an image from the validation-set.
        dir = coco.VAL_DIR
        filename = val_filenames[idx]
        captions = val_captions[idx]

    # Path for the image-file.
    path = os.path.join(dir, filename)

    # Print the captions for this image.
    x_label = ''
    for caption in captions:
        x_label += caption + '\n'

    # Load the image and show it
    img = coco.load_image(path)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel=x_label)
    plt.title(f'Filename : {filename}')
    plt.show()


show_image(idx=random.randint(0, 10000), train=True)

# Tokenizer
marker_start = 'ssss '
marker_end = ' eeee'

print('\nMarking training and val captions with \'{}\' and \'{}\'\n'.format(marker_start, marker_end))
train_captions_marked = mark_captions(captions_list=train_captions, mark_start=marker_start, mark_end=marker_end)
val_captions_marked = mark_captions(captions_list=val_captions, mark_start=marker_start, mark_end=marker_end)

train_captions_flat = flatten_captions(captions_list=train_captions_marked)
val_captions_flat = flatten_captions(captions_list=val_captions_marked)
caption_flat = train_captions_flat + val_captions_flat

tokenizer = TokenizerWrapper(
    texts=caption_flat,
    config=config
)
print(f'Total words in vocab : {tokenizer.num_words}')

train_tokens = tokenizer.captions_to_tokens(captions_list=train_captions_marked)
val_tokens = tokenizer.captions_to_tokens(captions_list=val_captions_marked)

# Testing marked captions and their respective tokens
print('Captions marked : {}'.format(print_list(train_captions_marked[0])))
print('Tokens of above marked captions : {}'.format(print_list(train_tokens[0])))

# Use pre-trained embedding layer GloVe and fine-tune it
glove = GloVe(tokenizer)

# Process the pre-trained VGG16 model
vgg16 = VGG16(
    batch_size=48,
    train_filenames=train_filenames,
    val_filenames=val_filenames,
    config=config
)

# Create the dataset we'll be using to train
# Dataset is a list with each item as a list of
# type : [transfer-value, caption (maybe tokenize)]
train_dataset = create_dataset_list(transfer_values=vgg16.transfer_values_train, captions=train_tokens)
val_dataset = create_dataset_list(transfer_values=vgg16.transfer_values_val, captions=val_tokens)

model = TrainModel(vgg16, glove, state_size=512)
model.decoder_model.summary()

model.decoder_model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks

callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=PATH_TRAINED_WEIGHTS,
    verbose=1,
)

callback_tensorboard = keras.callbacks.TensorBoard(
    log_dir=PATH_TENSORBOARD_LOGS,
    histogram_freq=1,
    write_graph=True,
    write_grads=True,
)

try:
    model.decoder_model.load_weights(PATH_TRAINED_WEIGHTS)
except Exception as error:
    print("Error trying to load checkpoint.")
    print('{}\n'.format(error))

# Calculating steps per epoch
num_captions_train = [len(captions) for captions in train_captions]
total_num_captions_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_captions_train / BATCH_SIZE)

model.decoder_model.fit_generator(
    # generator=,  # TODO : Complete and use COCOSequenceGenerator
    epochs=20,
    initial_epoch=0,
    steps_per_epoch=steps_per_epoch,
    callbacks=[callback_model_checkpoint, callback_tensorboard]
)
