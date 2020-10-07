from nlplingo.tasks.common.unary.base import UnaryDatapoint
from nlplingo.common.data_types import int_type, float_type
import numpy as np

from nlplingo.tasks.common.unary.util import _window_indices, _calculate_position_data
from nlplingo.tasks.common.feature.generator import FeatureGenerator

from abc import ABC, abstractmethod

class UnaryWithinSentence(UnaryDatapoint):

    def __init__(self, span, event_domain, embedding_vector_size,
                 label_str, sentence):
        super(UnaryWithinSentence, self).__init__(span, event_domain, embedding_vector_size, label_str)
        self.sentence = sentence
        self.span_token_indices = None # used in window, window vector, and position to compute appropriate values

    def sentence_word_embedding(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.sentence.word_embedding(max_length)

    def sentence_ner_type(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.sentence.ner_type(max_length)

    def sentence_word_embedding_vector(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.sentence.word_embedding_vector(max_length, self.embedding_vector_size)

    def unary_window_vector(self, neighbor_distance):
        """
        For use as a feature.

        We want to capture [word-on-left , target-word , word-on-right]
        Use self.lex_data to capture context window, each word's embeddings or embedding index
        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        :type max_sent_length: int
        :type neighbor_dist: int

        Returns:
            list[str]
        """
        # for lex_data, I want to capture: word-on-left target-word word-on-right
        rtn = np.zeros((2 * neighbor_distance + 1, self.embedding_vector_size), dtype=float_type)
        token_window = self._get_token_window(neighbor_distance)
        for (i, token) in token_window:
            if token.word_vector is not None:
                rtn[i, :] = token.word_vector
        return rtn


    def unary_window(self, neighbor_distance):
        """
        For use as a feature.

        We want to capture [word-on-left , target-word , word-on-right]
        Use self.lex_data to capture context window, each word's embeddings or embedding index
        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        :type max_sent_length: int
        :type neighbor_dist: int

        Returns:
            list[str]
        """
        rtn = np.zeros(2 * neighbor_distance + 1, dtype=int_type)
        # for lex_data, I want to capture: word-on-left target-word word-on-right
        token_window = self._get_token_window(neighbor_distance)
        for (i, token) in token_window:
            rtn[i] = token.vector_index
        return rtn

    def unary_word_position(self, max_sentence_length):
        """
        For use as a feature.
        :param max_sentence_length:
        :return:
        """
        rtn = np.zeros(max_sentence_length, dtype=int_type)
        rtn[:] = _calculate_position_data(self.span_token_indices, max_sentence_length)
        return rtn

    def _get_token_window(self, window_size):
        """
        +1
        Use _window_indices to generate window indices around self.span_token_indices.
        Any indices before the start or after the end of the entire sequence of tokens will be removed.
        :type window_size: int
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        for i, w in enumerate(_window_indices(self.span_token_indices, window_size)):
            if w < 0 or w >= len(self.sentence.tokens):
                continue
            ret.append((i, self.sentence.tokens[w]))
        return ret

class UnaryWithinSentenceFeatureGenerator(FeatureGenerator, ABC):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(UnaryWithinSentenceFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    @abstractmethod
    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        if hasattr(self.hyper_params, 'max_sentence_length'):
            self.assign_example(example, "sentence_word_embedding", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "sentence_ner_type", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "sentence_word_embedding_vector", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "unary_word_position", [self.hyper_params.max_sentence_length])

        if hasattr(self.hyper_params, 'neighbor_distance'):
            self.assign_example(example, "unary_window_vector", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "unary_window", [self.hyper_params.neighbor_distance])
