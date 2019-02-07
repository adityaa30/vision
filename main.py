import image_captioning.coco as coco
from image_captioning.models.models import VGG16
from image_captioning.tokenizer import *
from image_captioning.glove import GloVe
from image_captioning.utils import *

import random
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Constants
STATE_SIZE = 512
BATCH_SIZE = 128
PATH_TRAINED_WEIGHTS = 'trained_weights/model-1.ckpt'
PATH_TENSORBOARD_LOGS = 'tensorboard/logs-1/'

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
    plt.title('Filename : {}'.format(filename))
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
    texts=caption_flat
)

train_tokens = tokenizer.captions_to_tokens(captions_list=train_captions_marked)
val_tokens = tokenizer.captions_to_tokens(captions_list=val_captions_marked)

# Testing marked captions and their respective tokens
print('Captions marked : {}'.format(print_list(train_captions_marked[0])))
print('Tokens of above marked captions : {}'.format(print_list(train_tokens[0])))


def get_random_captions_tokens(index):
    """
    Generated a list of token sequences, each token sequence
    is selected randomly from the respective captions for
    a particular image

    :param index: list of id's for a set of images
    :return: list of randomly selected token sequences
    """
    token_sequences = []
    for index in index:
        token_index = np.random.choice(len(train_tokens[index]))
        token_sequence = train_tokens[index][token_index]
        token_sequences.append(token_sequence)

    return token_sequences


# Use pre-trained embedding layer GloVe and fine-tune it
glove = GloVe(tokenizer)

# Process the pre-trained VGG16 model
vgg16 = VGG16(
    batch_size=32,
    train_filenames=train_filenames,
    val_filenames=val_filenames
)

# GRU network

transfer_values_input = keras.layers.Input(shape=(vgg16.image_process.transfer_values_size,))
decoder_transfer_map = keras.layers.Dense(
    units=STATE_SIZE,
    activation='tanh',
    name='decoder_transfer_map'
)

decoder_input = keras.layers.Input(shape=(None,), name='decoder_input')

decoder_embedding = keras.layers.Embedding(
    input_dim=glove.num_words,
    output_dim=glove.embedding_dimension,
    weights=[glove.embedding_matrix],
    trainable=True,
    name='decoder_embedding'
)

decoder_gru1 = keras.layers.CuDNNGRU(
    units=STATE_SIZE,
    return_sequences=True,
    name='decoder_lstm1'
)

decoder_gru2 = keras.layers.CuDNNGRU(
    units=STATE_SIZE,
    return_sequences=True,
    name='decoder_lstm2'
)

decoder_gru3 = keras.layers.CuDNNGRU(
    units=STATE_SIZE,
    return_sequences=True,
    name='decoder_lstm3'
)

decoder_dense = keras.layers.Dense(
    units=glove.num_words,
    activation='linear',
    name='decoder_output'
)


def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)

    net = decoder_input

    net = decoder_embedding(net)

    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    _decoder_output = decoder_dense(net)

    return _decoder_output


decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = keras.models.Model(
    inputs=[transfer_values_input, decoder_input],
    outputs=[decoder_output]
)

decoder_model.summary()

# decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

decoder_model.compile(
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
    write_images=True
)

try:
    decoder_model.load_weights(PATH_TRAINED_WEIGHTS)
except Exception as error:
    print("Error trying to load checkpoint.")
    print('{}\n'.format(error))

# Calculating steps per epoch
num_captions_train = [len(captions) for captions in train_captions]
total_num_captions_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_captions_train / BATCH_SIZE)


# Create a Sequence model

class COCOSequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __getitem__(self, index):
        """
        Gets batch at position index
        """

        pass

    def __len__(self):
        """
        :return: Number of batch in the Sequence
        """
        return math.ceil(len(self.x) / self.batch_size)


decoder_model.fit_generator(
    # generator=,  # TODO
    epochs=20,
    initial_epoch=0,
    steps_per_epoch=steps_per_epoch,
    callbacks=[callback_model_checkpoint, callback_tensorboard]
)
