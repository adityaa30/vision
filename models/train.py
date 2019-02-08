import keras
import math
from keras import backend as K


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

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __getitem__(self, idx):
        """
        Gets batch at position index
        """

        pass

    def __len__(self):
        """
        :return: Number of batch in the Sequence
        """
        return math.ceil(len(self.x) / self.batch_size)
