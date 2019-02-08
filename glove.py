from tokenizer import TokenizerWrapper

import pickle
import bcolz as bcolz
import numpy as np
import pandas as pd
import csv
import os


class GloVe:
    DATASET_DIR = 'dataset/'
    PATH_GLOVE_MATRIX = DATASET_DIR + 'glove.6B.300d.txt'
    PATH_EMBEDDING_MATRIX = DATASET_DIR + 'embedding_matrix.npy'

    # PATH_BCOLZ_GLOVE = DATASET_DIR + 'glove.6B.300d.dat'

    def __init__(self, tokenizer):
        """
        :param tokenizer: instance of TokenizerWrapper class consisting of vocab
        """
        self.tokenizer = tokenizer
        assert isinstance(tokenizer, TokenizerWrapper) and tokenizer.index_word is not None
        self.embedding_dimension = 300
        self.threshold_value = 3
        self.embedding_matrix = None

        if os.path.exists(self.PATH_EMBEDDING_MATRIX):
            print(f'\nLoading pre-processed embedding matrix from {self.PATH_EMBEDDING_MATRIX}')
            self.embedding_matrix = np.load(self.PATH_EMBEDDING_MATRIX)
            self.num_words = self.tokenizer.num_words
            # self.glove_vectors = self.load_glove_vectors(self.DATASET_DIR, 300)

        else:
            print('\nProcessing glove vectors to create embedding matrix.')
            # self.process_glove(self.DATASET_DIR, 300)
            self.glove_vectors = None  # finally it will  be converted to a DataFrame containing all the GloVe vectors
            self.load_glove_vectors()
            self.num_words = self.tokenizer.num_words
            self.embedding_matrix = np.random.rand(
                self.num_words,
                self.embedding_dimension
            )
            self.prepare_embedding_matrix()

    # def process_glove(self, glove_path, dims):
    #     '''
    #     This method writes 2 pkl files that store the word list from the glove
    #     vectors (~400K words) and a corresponding file that stores each words index
    #
    #     :param glove_path: path to glove file
    #     :param dims: the dimentions to use
    #
    #     this makes the total path as {glove_path}/glove.6B.{dims}d.txt
    #
    #     NOTE: bcolz stores a .dat folder that is used for compressing the large number of
    #     words that are present in the glove file
    #
    #     Run this function once to generate the respective pickle files.
    #
    #     '''
    #     path = os.path.join(glove_path, 'glove.6B.{dims}d.txt'.format(dims=dims))
    #     dat_file_name = os.path.join(glove_path, 'glove.6B.{dims}d.dat'.format(dims=dims))
    #     print(path)
    #     words = []
    #     idx = 0
    #     word2idx = {}
    #     vectors = bcolz.carray(np.zeros(1), rootdir=dat_file_name, mode='w')
    #
    #     with open(path, 'rb') as f:
    #         for l in f:
    #             line = l.decode().split()
    #             word = line[0]
    #             words.append(word)
    #             word2idx[word] = idx
    #             idx += 1
    #             vect = np.array(line[1:]).astype(np.float)
    #             vectors.append(vect)
    #
    #     vectors = bcolz.carray(vectors[1:].reshape((400000, dims)), rootdir=dat_file_name, mode='w')
    #     vectors.flush()
    #     pickle.dump(words, open(f'{glove_path}/6B.{dims}_words.pkl', 'wb'))
    #     pickle.dump(word2idx, open(f'{glove_path}/6B.{dims}_idx.pkl', 'wb'))
    #
    # def load_glove_vectors(self, path, dims):
    #     glove_path = path
    #     vectors = bcolz.open(os.path.join(glove_path, f'glove.6B.{dims}d.dat'))[:]
    #     words = pickle.load(open(os.path.join(glove_path, f'6B.{dims}_words.pkl'), 'rb'))
    #     word2idx = pickle.load(open(os.path.join(glove_path, f'6B.{dims}_idx.pkl'), 'rb'))
    #     glove = {w: vectors[word2idx[w]] for w in words}
    #     return glove

    def load_glove_vectors(self):
        print('Loading glove word vectors..')

        self.glove_vectors = pd.read_table(
            self.PATH_GLOVE_MATRIX,
            sep=" ",
            index_col=0,
            header=None,
            quoting=csv.QUOTE_NONE
        )

        print('Found {} word vectors.'.format(self.glove_vectors.shape))

    def get_glove_vector(self, word):
        """
        :param word: string
        :return: n-dimensional glove word-vector corresponding to @word else None
        """
        assert isinstance(self.glove_vectors, pd.DataFrame)
        if word in self.glove_vectors.index:
            return self.glove_vectors.loc[word].values
        return None

    def find_word_closest_in_tokenizer(self, word):
        """
        Find nearest word in the glove vectors to the given @word which is present
        in the Tokenizer and has frequency above threshold value

        :param word: string
        :return: n-dimensional glove vector of the given word
        """
        assert isinstance(self.glove_vectors, pd.DataFrame)
        difference = self.glove_vectors.values - self.get_glove_vector(word)
        delta = np.sum(difference * difference, axis=1)

        # Get the indices sorted according to minimum distance
        sorted_indices = np.argsort(delta, kind='quicksort')
        for index in sorted_indices:
            current_word = self.glove_vectors.iloc[index].name
            frequency = self.tokenizer.word_counts.get(current_word)
            if frequency is not None and frequency > self.threshold_value:
                print(f'-Placed embedding of word \'{current_word}\' to \'{word}\'')
                return self.get_glove_vector(current_word)
            else:
                continue

        # Word does not exist in the glove vectors or tokenizer
        # Assign random vector
        return None

    def prepare_embedding_matrix(self):
        print('\nPreparing embedding matrix..')

        for word, i in self.tokenizer.word_index.items():

            embedding_vector = self.get_glove_vector(word)
            if self.tokenizer.word_counts[word] <= self.threshold_value and embedding_vector is not None:
                # Very less occurrence of this word
                # Give embedding vector with more frequent word with closer semantic meaning
                embedding_vector = self.find_word_closest_in_tokenizer(word)

            if embedding_vector is not None:
                # words not found in the embedding index will be all random.
                self.embedding_matrix[i - 1] = embedding_vector
            print('Processed word : \'{}\'\t\ttoken id :{}'.format(word, i))

            # Save the embedding matrix after each word is processed
            if i % 1000 == 0:
                np.save(self.PATH_EMBEDDING_MATRIX, self.embedding_matrix)

        np.save(self.PATH_EMBEDDING_MATRIX, self.embedding_matrix)
        print('Embedding matrix of shape {} prepared.'.format(self.embedding_matrix.shape))


# Testing purpose only
if __name__ == '__main__':
    pass
