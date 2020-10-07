
from nlplingo.tasks.common.binary.binary_event_event import BinaryEventEvent

class EventEventRelationExample(BinaryEventEvent):
    def __init__(self, arg0, arg1, event_domain, label_str, serif_sentence=None, serif_event_0=None, serif_event_1=None):
    # def __init__(self, anchor1, anchor2, sentence, event_domain, extractor_params, features, eer_type=None, usable_features=None):
        if label_str is not None:
            if 'Negative' in label_str:
                label_str = 'NA'
        super(EventEventRelationExample, self).__init__(arg0, arg1, event_domain, label_str)
        self.sentence = self.arg0.sentence # this is a temporary hack; it should be removed
        self.anchor1 = self.arg0.span # this is a temporary hack; it should be removed
        self.anchor2 = self.arg1.span # this is a temporary hack; it should be removed

        self.serif_sentence = serif_sentence
        self.serif_event_0 = serif_event_0
        self.serif_event_1 = serif_event_1

        self.joint_serif_prop_tree = None # A tree containing the two Serif events

    @property
    def eer_type(self):
        """:rtype: str"""
        return self.label_str

    @eer_type.setter
    def eer_type(self, label):
        """:type label: str"""
        self.label_str = label

    def get_eer_type_index(self):
        return self.event_domain.get_eer_type_index(self.eer_type)

    def to_triplet_with_relation(self):
        # This can only be used for within-sentence relations.
        triplet = self.to_triplet()
        triplet.update({'relation' : self.event_domain.get_eer_type_from_index(int(self.label[0]))})
        return triplet