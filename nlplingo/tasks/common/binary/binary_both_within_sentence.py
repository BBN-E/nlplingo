from nlplingo.tasks.common.binary.base import BinaryDatapoint
from nlplingo.tasks.common.feature.generator import FeatureGenerator
import numpy as np

from nlplingo.common.data_types import int_type
from nlplingo.tasks.common.unary.util import _window_indices
from itertools import chain

from abc import ABC, abstractmethod

class BinaryBothWithinSentence(BinaryDatapoint):

    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type arg0: nlplingo.tasks.common.unary.within_sentence.UnaryWithinSentence
        :type arg1: nlplingo.tasks.common.unary.within_sentence.UnaryWithinSentence
        """
        super(BinaryBothWithinSentence, self).__init__(arg0, arg1, event_domain, label_str)

    def arg0_sentence_word_embedding(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.arg0.sentence_word_embedding(max_length)

    def arg0_sentence_ner_type(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.arg0.sentence_ner_type(max_length)

    def arg0_window_vector(self, neighbor_distance):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        return self.arg0.unary_window_vector(neighbor_distance)

    def arg1_window_vector(self, neighbor_distance):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        return self.arg1.unary_window_vector(neighbor_distance)

    def arg1_window(self, neighbor_distance):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        return self.arg1.unary_window(neighbor_distance)

    def arg0_word_position(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.arg0.unary_word_position(max_length)

    def arg1_word_position(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.arg1.unary_word_position(max_length)

    def arg0_arg1_window_vector(self, neighbor_distance):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        return np.concatenate((self.arg0_window_vector(neighbor_distance), self.arg1_window_vector(neighbor_distance)), axis=0)

    def _get_token_windows(self, tokens, window_size):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type window_size: int
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(chain(_window_indices(self.arg0.span_token_indices, window_size), _window_indices(self.arg1.span_token_indices, window_size))):
            if w < 0 or w >= len(tokens):
                continue
            ret.append((i, tokens[w]))
        return ret


    def arg0_arg1_window(self, neighbor_distance, embeddings_data):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        window_size = 2*neighbor_distance + 1
        rtn = np.zeros(2*window_size,  dtype=int_type)

        # TODO: not a good notation; fix it
        if 'none_token_index' in embeddings_data:
            none_token_index = embeddings_data['none_token_index']
        else:
            none_token_index = 1
        rtn[:] = none_token_index

        #token_windows = self._get_token_windows(self.arg0.sentence.tokens, window_size)
        token_windows = self._get_token_windows(self.arg0.sentence.tokens, neighbor_distance)
        for (i, token) in token_windows:
            rtn[i] = token.vector_index  # local window around arg1 and arg2

        return rtn

    def arg0_arg1_window_modern(self, neighbor_distance):
        """
        For use as a feature.
        :param neighbor_distance:
        :return:
        """
        return np.concatenate((self.arg0.unary_window(neighbor_distance), self.arg1_window(neighbor_distance)), axis=0)

    def to_triplet(self):
        # This can only be used for within-sentence relations.
        anchor1_index_start = self.arg0.span.start_char_offset() - self.arg0.sentence.start_char_offset()
        anchor1_index_end = self.arg0.span.end_char_offset() - self.arg0.sentence.start_char_offset()
        anchor2_index_start = self.arg1.span.start_char_offset() - self.arg0.sentence.start_char_offset()
        anchor2_index_end = self.arg1.span.end_char_offset() - self.arg0.sentence.start_char_offset()
        assert(anchor1_index_start >= 0 and anchor2_index_start >= 0 and anchor1_index_end >= 0 and anchor2_index_end >= 0)
        return {'text': self.arg0.sentence.text, 'h' : {'pos' : (anchor1_index_start, anchor1_index_end)}, 't' : {'pos' : (anchor2_index_start, anchor2_index_end)}}

class BinaryBothWithinSentenceFeatureGenerator(FeatureGenerator, ABC):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(BinaryBothWithinSentenceFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    @abstractmethod
    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        if hasattr(self.hyper_params, 'max_sentence_length'):
            self.assign_example(example, "arg0_sentence_word_embedding", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "arg0_word_position", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "arg1_word_position", [self.hyper_params.max_sentence_length])
            self.assign_example(example, "arg0_sentence_ner_type", [self.hyper_params.max_sentence_length])

        if hasattr(self.hyper_params, 'neighbor_distance'):
            self.assign_example(example, "arg0_window_vector", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "arg1_window_vector", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "arg1_window", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "arg0_arg1_window_vector", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "arg0_arg1_window_modern", [self.hyper_params.neighbor_distance])
            self.assign_example(example, "arg0_arg1_window", [self.hyper_params.neighbor_distance, self.extractor_params.get('embeddings')])
