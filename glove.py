from image_captioning.tokenizer import TokenizerWrapper
import numpy as np


class GloVe:
    PATH_TRAINED_WEIGHTS = 'dataset/glove.6B.300d.txt'

    def __init__(self, tokenizer, num_words):
        """
        :param tokenizer: instance of TokenizerWrapper class consisting of vocab
        :param num_words: maximum number of words in the vocab
        """
        self.tokenizer = tokenizer
        assert isinstance(tokenizer, TokenizerWrapper) and tokenizer.index_word is not None
        self.num_words = num_words
        self.embedding_dimension = 300

        self.embeddings_index = {}
        self.load_embeddings()

        self.embedding_matrix = np.random.rand(
            self.num_words,
            self.EMBEDDING_DIMENSION
        )
        self.prepare_embedding_matrix()

    def load_embeddings(self):
        print('Indexing word vectors..')

        with open(self.PATH_TRAINED_WEIGHTS) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefficients = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefficients

        print('Found {} word vectors.'.format(len(self.embeddings_index)))

    def prepare_embedding_matrix(self):
        print('Preparing embedding matrix..')

        for word, i in self.tokenizer.word_index.items():
            if i > self.num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in the embedding index will be all random.
                self.embedding_matrix[i] = embedding_vector

        print('Embedding matrix of shape {} prepared.'.format(self.embedding_matrix.shape))
