
from nlplingo.tasks.common.binary.base import BinaryDatapoint

pair_labels = {'DIFFERENT': 0, 'SAME': 1}


# +1
class EventPairExample(BinaryDatapoint):
    def __init__(self, eg1, eg2, label_string):
        """
        :type eg1: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        :type eg2: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        :type label_string: 'SAME' or 'DIFFERENT'
        """
        super(EventPairExample, self).__init__(
            eg1, eg2, None, None, label_string, None)

        self.label = pair_labels[label_string]
        self.trigger_window_vector1 = eg1.window_data_vector
        self.trigger_window_vector2 = eg2.window_data_vector

    @property
    def eg1(self):
        """:rtype: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        return self.left_datapoint

    @eg1.setter
    def eg1(self, trigger_candidate):
        """:type trigger_candidate: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        self.left_datapoint = trigger_candidate

    @property
    def eg2(self):
        """:rtype: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        return self.right_datapoint

    @eg2.setter
    def eg2(self, trigger_candidate):
        """:type trigger_candidate: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        self.right_datapoint = trigger_candidate

    @property
    def label_string(self):
        """:rtype: str"""
        return self.label_str

    @label_string.setter
    def label_string(self, label):
        """:type label: str"""
        self.label_str = label

    def _allocate_arrays(self, hyper_params, embedding_size, none_token_index, features):
        return

    def to_data_dict(self, features):
        """
        :type features:  nlplingo.tasks.eventpair.feature.EventPairFeature
        :rtype: dict[str:numpy.ndarray]
        """
        d = dict()

        if self.trigger_window_vector1 is not None:
            d[features.c_trigger_window_vector1] = self.trigger_window_vector1

        if self.trigger_window_vector2 is not None:
            d[features.c_trigger_window_vector2] = self.trigger_window_vector2

        d['label'] = self.label

        return d
