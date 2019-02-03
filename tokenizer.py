from keras_preprocessing.text import Tokenizer


class TokenizerWrapper(Tokenizer):
    def __init__(self, texts, num_words=None):
        """
        :param texts: lists of strings in the data-set
        :param num_words: max number of words to user
        """
        Tokenizer.__init__(self, num_words=num_words)

        self.fit_on_texts(texts)

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

    def captions_to_tokens(self, captions_list):
        """
        Convert a @captions_list to
        a list-of-list of integer-tokens.
        :param captions_list: list of lists with text-captions
        """

        # text_to_sequences() takes a list of texts
        tokens = [self.texts_to_sequences(captions)
                  for captions in captions_list]

        return tokens
