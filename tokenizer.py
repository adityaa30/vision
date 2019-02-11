from keras_preprocessing.text import Tokenizer
import pandas as pd
import bcolz
from config import Config


class TokenizerWrapper(Tokenizer):
    def __init__(self, texts, config):
        """
        :param texts: lists of strings in the data-set
        :param config: Instance of Config class
        """
        Tokenizer.__init__(self)
        assert isinstance(config, Config)

        self.config = config
        self.fit_on_texts(texts)
        self.num_words = len(self.word_index.keys())
        self.word_counts_sorted = pd.DataFrame(sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True))

    def token_to_word(self, token):
        """
        Look-up a single word from @token

        :param token: Integer token of a word
        :return: corresponding word of @token
        """

        word = " " if token == 0 else self.index_word[token]
        return word

    def tokens_to_string(self, tokens):
        """
        Convert the given list of tokens into the respective
        sentence

        :param tokens: list of integer token (of a word)
        :return: corresponding sentence of @tokens
        """

        words = [self.index_word[token]
                 for token in tokens
                 if token != 0]

        text = " ".join(words)
        return text

    def captions_to_tokens(self, captions_list, train=True):
        """
        Convert a @captions_list to
        a list-of-list of integer-tokens.

        :param captions_list: list of lists with text-captions
        :param train:
                True if captions are from training set
                False if captions are from cross-validations set
        """

        # text_to_sequences() takes a list of texts
        tokens = [self.texts_to_sequences(captions)
                  for captions in captions_list]

        if train:
            return bcolz.carray(tokens, rootdir=self.config.paths.BCOLZ_TRAIN_CAPTIONS, mode='w')
        else:
            return bcolz.carray(tokens, rootdir=self.config.paths.BCOLZ_VAL_CAPTIONS, mode='w')
