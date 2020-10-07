
import numpy as np

from nlplingo.tasks.common.binary.binary_event_entity import BinaryEventEntity
from nlplingo.common.data_types import int_type

class EventArgumentExample(BinaryEventEntity):

    def __init__(self, arg0, arg1, event_domain, label_str):
    # def __init__(self, anchor, argument, sentence, event_domain, extractor_params, features, hyper_params, event_role=None, usable_features=None):
        """We are given an anchor, candidate argument, sentence as context, and a role label (absent in decoding)
        :type anchor: nlplingo.text.text_span.Anchor
        :type argument: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type extractor_params: dict
        :type features: nlplingo.tasks.eventargument.feature.EventArgumentFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type event_role: str
        """

        super(EventArgumentExample, self).__init__(arg0, arg1, event_domain, label_str)
        num_labels = len(self.event_domain.event_roles)
        self.label = np.zeros(num_labels, dtype=int_type)

        # vec_size = extractor_params['embeddings']['vector_size']
        # anchor_datapoint = EventDatapoint(
        #    anchor, event_domain, vec_size, anchor.label, usable_features)
        #argument_datapoint = EntityDatapoint(
        #    argument, event_domain, vec_size, argument.label, usable_features)

        #super(EventArgumentExample, self).__init__(
        #    anchor_datapoint, argument_datapoint,
        #    event_domain, features, event_role, usable_features)

        # self.sentence = sentence
        # self.anchor_obj = None
        # if 'none_token_index' in extractor_params['embeddings']:
        #    none_token_index = extractor_params['embeddings']['none_token_index']
        # else:
        #    none_token_index = 1
        #self._allocate_arrays(hyper_params,
        #                      extractor_params['embeddings']['vector_size'],
        #                      none_token_index,
        #                      features)

    @property
    def event_role(self):
        """:rtype: str"""
        return self.label_str

    @event_role.setter
    def event_role(self, label):
        """:type label: str"""
        self.label_str = label

    @property
    def anchor(self):
        """:rtype: nlplingo.text.text_span.Anchor"""
        return self.arg0.span

    @property
    def argument(self):
        """:rtype: nlplingo.text.text_span.EventArgument"""
        return self.arg1.span

    """
    @argument.setter
    def argument(self, argument):
        :type argument: nlplingo.text.text_span.EventArgument
        argument_datapoint = EntityDatapoint(
            argument, self.event_domain, self.argument.embedding_vector_size,
            argument.label, usable_features)
        self.right_datapoint = argument_datapoint
    """

    def get_event_role_index(self):
        """
        +1
        """
        return self.event_domain.get_event_role_index(self.event_role)

    def to_triplet_with_relation(self):
        # This can only be used for within-sentence relations.
        triplet = self.to_triplet()
        triplet.update({'relation' : self.event_role})
        return triplet