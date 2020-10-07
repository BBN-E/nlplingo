from nlplingo.common.utils import Struct
from nlplingo.tasks.common.unary.within_sentence import UnaryWithinSentence, UnaryWithinSentenceFeatureGenerator
import numpy as np
from nlplingo.common.data_types import int_type
from abc import ABC, abstractmethod

class EventWithinSentence(UnaryWithinSentence):

    def __init__(self, span, event_domain, embedding_vector_size,
                 label_str, sentence, head_only=False):
        super(EventWithinSentence, self).__init__(span, event_domain, embedding_vector_size, label_str, sentence)
        if not head_only:
            if self.span.tokens is not None:
                self.span_token_indices = Struct(start=self.span.tokens[0].index_in_sentence,
                                              end=self.span.tokens[-1].index_in_sentence, head=self.span.head().index_in_sentence)
        else:
            head_token_index = span.head().index_in_sentence
            self.span_token_indices = Struct(start=head_token_index, end=head_token_index, head=head_token_index)

    def event_embeddings(self, max_length):
        """
        Return a numpy array (with shape equal to the length of the sentence) filled with the event type label.
        :param max_length:
        :return:
        """
        rtn = np.zeros(max_length, dtype=int_type)
        rtn[:] = self.event_domain.get_event_type_index('None')
        for i, token in enumerate(self.sentence.tokens):
            # in some usages, this used to exclude the None event type
            rtn[i] = self.event_domain.get_event_type_index(self.event_type)
        return rtn

    @property
    def event_type(self):
        """:rtype: set[str]"""
        return self.label_str

    @event_type.setter
    def event_type(self, event_type):
        """:type event_type: set[str]"""
        self.label_str = event_type

    @property
    def anchor(self):
        """:rtype: nlplingo.text.text_span.Anchor"""
        return self.span

    @anchor.setter
    def anchor(self, anchor):
        """:type anchor: nlplingo.text.text_span.Anchor"""
        self.span = anchor

class EventWithinSentenceFeatureGenerator(UnaryWithinSentenceFeatureGenerator, ABC):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EventWithinSentenceFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    @abstractmethod
    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        super(EventWithinSentenceFeatureGenerator, self).populate_example(example)
        if hasattr(self.hyper_params, 'max_sentence_length'):
            self.assign_example(example, "event_embeddings", [self.hyper_params.max_sentence_length])