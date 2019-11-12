import os
import pickle
import logging
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from vision.coco import COCODataset

K = tf.keras
KL = tf.keras.layers
KA = tf.keras.applications

LOGGER = logging.getLogger(__name__)


class MobileNetV2:
    SAVE_PATH = os.path.join('dataset', 'transfer_vals')
    TRAIN_PATH = os.path.join(SAVE_PATH, 'train')
    VAL_PATH = os.path.join(SAVE_PATH, 'val')

    def __init__(self):
        self._model = KA.mobilenet_v2.MobileNetV2(
            include_top=False, weights='imagenet')

        self.input_shape = (224, 224, 3)
        self.output_shape = (7, 7, 1280)

        self.transfer_model = tf.keras.Model(
            self._model.input,
            self._model.layers[-1].output
        )

        self.dataset = COCODataset(train_model='mobilenetv2')

    def save_path(self, train=True):
        return self.TRAIN_PATH if train else self.VAL_PATH

    def decode_transfer_vals(self, train=True):
        LOGGER.log('Starting to decode transfer values...')

        i = 0
        dataset = self.dataset.get_dataset(train)
        for img, path in dataset:
            batch_features = self.transfer_model(img)
            batch_features = tf.reshape(
                batch_features,
                (batch_features.shape[0], -1, batch_features.shape[3])
            )

            i += 1
            if i % 100 == 0:
                print(f'MobileNetV2 iterations: {i}')

            for bf, p in zip(batch_features, path):
                path_of_feature = os.path.join(
                    self.save_path(train),
                    os.path.basename(p.numpy().decode('utf-8'))
                )
                np.save(path_of_feature, bf.numpy())

        LOGGER.log('Finished decoding transfer values')


# Ref: https://www.tensorflow.org/beta/tutorials/text/image_captioning
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = KL.Dense(units)
        self.W2 = KL.Dense(units)
        self.V = KL.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNNEncoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNNEncoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = KL.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNNDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units

        self.embedding = KL.Embedding(vocab_size, embedding_dim)
        self.gru = KL.GRU(self.units,
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')
        self.fc1 = KL.Dense(self.units)
        self.fc2 = KL.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
