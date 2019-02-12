import keras
from keras import backend as K

import math
import numpy as np
import itertools


class TrainModel:
    def __init__(self, transfer_model, glove, state_size):
        self.transfer_model = transfer_model
        self.glove = glove
        self.state_size = state_size

        self.transfer_values_input = keras.layers.Input(shape=(self.transfer_model.image_process.transfer_values_size,))
        self.decoder_transfer_map = keras.layers.Dense(
            units=self.state_size,
            activation='tanh',
            name='decoder_transfer_map'
        )

        self.decoder_input = keras.layers.Input(shape=(None,), name='decoder_input')

        with K.name_scope('embedding'):
            self.decoder_embedding = keras.layers.Embedding(
                input_dim=self.glove.num_words,
                output_dim=self.glove.embedding_dimension,
                weights=[self.glove.embedding_matrix],
                trainable=False,
                name='decoder_embedding'
            )

        with K.name_scope('gru1'):
            self.decoder_gru1 = keras.layers.CuDNNGRU(
                units=self.state_size,
                return_sequences=True,
                name='decoder_gru1'
            )

        with K.name_scope('gru2'):
            self.decoder_gru2 = keras.layers.CuDNNGRU(
                units=self.state_size,
                return_sequences=True,
                name='decoder_gru2'
            )

        with K.name_scope('gru2'):
            self.decoder_gru3 = keras.layers.CuDNNGRU(
                units=self.state_size,
                return_sequences=True,
                name='decoder_gru3'
            )

        with K.name_scope('fc'):
            self.decoder_dense = keras.layers.Dense(
                units=self.glove.num_words,
                activation='linear',
                name='decoder_output'
            )

        self.decoder_output = self.connect_decoder(transfer_values=self.transfer_values_input)

        self.decoder_model = keras.models.Model(
            inputs=[self.transfer_values_input, self.decoder_input],
            outputs=[self.decoder_output]
        )

    def connect_decoder(self, transfer_values):
        initial_state = self.decoder_transfer_map(transfer_values)

        net = self.decoder_input

        net = self.decoder_embedding(net)

        net = self.decoder_gru1(net, initial_state=initial_state)
        net = self.decoder_gru2(net, initial_state=initial_state)
        net = self.decoder_gru3(net, initial_state=initial_state)

        _decoder_output = self.decoder_dense(net)

        return _decoder_output


class COCOSequenceGenerator(keras.utils.Sequence):

    def __init__(self, model, config, train=True):
        """
        Constructor to initialize the values of the dataset

        :param model: Model class from models/models.py to fetch dataset
        :param config: Instance of Config class
        :param train:
                True if dataset is used for Training data
                False if dataset is used for Cross-validation data
        """
        self.model = model
        self.batch_size = config.TRAIN_BATCH_SIZE
        self.config = config

        if train:
            self.dataset = self.model.train_dataset
            self.transfer_values = self.model.transfer_values_train
            self.captions = self.model.train_tokens
        else:
            self.dataset = self.model.val_dataset
            self.transfer_values = self.model.transfer_values_val
            self.captions = self.model.val_tokens

        # Shuffle the dataset
        self.transfer_values_idx = None
        self.captions_idx = None
        self.shuffle_dataset()

    def __getitem__(self, idx):
        """
        Gets batch at position index
        """
        batch_transfer_values_idx = np.array(
            self.transfer_values_idx[idx * self.batch_size:(idx + 1) * self.batch_size])

        batch_captions_idx = np.array(self.captions_idx[idx * self.batch_size:(idx + 1) * self.batch_size])

        batch_transfer_values = np.array([])
        batch_captions = np.array([])
        for i, idx in enumerate(batch_transfer_values_idx, start=0):
            batch_transfer_values = np.r_[batch_transfer_values, self.transfer_values[idx]]
            batch_captions = np.r_[batch_captions, self.captions[idx][batch_captions_idx[i]]]

        batch_captions = itertools.zip_longest(batch_captions, fillvalue=self.config.PADDING_FILL_VALUE)

        decoder_input_data = batch_captions[:, 0:-1]
        decoder_output_data = batch_captions[:, 1:]

        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': batch_transfer_values
        }

        # Dict for the output-data.
        y_data = {
            'decoder_output': decoder_output_data
        }

        return x_data, y_data

    def __len__(self):
        """
        :return: Number of batches in the Sequence
        """
        return math.ceil(len(self.transfer_values) / self.batch_size)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.shuffle_dataset()

    def shuffle_dataset(self):
        np.random.shuffle(self.dataset)
        self.transfer_values_idx = self.dataset[:, 0]
        self.captions_idx = self.dataset[:, 1]
