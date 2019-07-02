import tensorflow as tf
import pandas as pd


class TokenizerWrapper(tf.keras.preprocessing.text.Tokenizer):
    UNK = '<unk>'
    START = '<start>'
    END = '<end>'
    PAD = '<pad>'

    def __init__(self, texts, num_words=10000):
        """
        :param texts: lists of strings in the data-set
        """
        tf.keras.preprocessing.text.Tokenizer.__init__(
            self, num_words=num_words,
            oov_token="<unk>",
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
        )

        self.fit_on_texts(texts)
        self.word_index[self.PAD] = 0
        self.index_word[0] = self.PAD

    @staticmethod
    def calc_max_length(tensor):
        # Find the maximum length of any caption in our dataset
        return max(len(t) for t in tensor)

    def get_padded_sequences(self, texts):
        """
        Converts the given texts into their respective sequences.
        All the sequences are furthure padded
        """
        seqs = self.texts_to_sequences(texts)
        pad_seqs = tf.keras.preprocessing.sequence.pad_sequences(
            seqs, padding='post')

        max_length = self.calc_max_length(pad_seqs)

        return pad_seqs, max_length