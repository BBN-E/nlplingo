
from nlplingo.tasks.common.binary.binary_entity_entity import BinaryEntityEntity
from nlplingo.common.data_types import int_type
import numpy as np

class EntityRelationExample(BinaryEntityEntity):
    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type span1: nlplingo.text.text_span.Anchor
        :type span2: nlplingo.text.text_span.Anchor
        :type label: int            # either 0 or 1
        :type span1_sentence: nlplingo.text.text_span.Sentence
        :type span2_sentence: nlplingo.text.text_span.Sentence
        :type extractor_params: dict
        :type features: nlplingo.tasks.eventcoref.feature.EventCorefFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        if label_str is not None:
            if 'Negative' in label_str:
                label_str = 'NA'
        super(EntityRelationExample, self).__init__(arg0, arg1, event_domain, label_str)
        # generate label data
        num_labels = len(self.event_domain.entity_relation_types)
        self.label = np.zeros(num_labels, dtype=int_type)

    @property
    def relation_type(self):
        """:rtype: str"""
        return self.label_str

    @relation_type.setter
    def relation_type(self, label):
        """:type label: str"""
        self.label_str = label

    def get_relation_type_index(self):
        return self.event_domain.get_entity_relation_type_index(self.relation_type)

    def to_triplet_with_relation(self):
        # This can only be used for within-sentence relations.
        triplet = self.to_triplet()
        triplet.update({'relation' : self.relation_type})
        return triplet