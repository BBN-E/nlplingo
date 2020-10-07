import numpy as np

from nlplingo.tasks.common.binary.base import BinaryDatapoint
from nlplingo.tasks.common.unary.base import UnaryDatapoint

pair_labels = {'DIFFERENT': 0, 'SAME': 1}


class EventFrameExample(UnaryDatapoint):
    def __init__(self, trigger_example, argument_examples, extractor_params, hyper_params):
        """
        :param trigger_example: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint
        :param argument_examples: list[nlplingo.tasks.eventargument.example.EventArgumentExample]
        """
        # going to assume for now, that each role has max 1 argument
        arg_dict = {arg.event_role.lower(): arg for arg in argument_examples}
        super(EventFrameExample, self).__init__(
            (trigger_example, arg_dict), None, None, None, None)

        self._allocate_arrays(hyper_params,
                              extractor_params['embeddings']['vector_size'],
                              None,
                              None)

    @property
    def trigger_example(self):
        """:rtype: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        return self.span[0]

    @trigger_example.setter
    def trigger_example(self, candidate):
        """:type candidate: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
        self.span = (candidate, self.argument_examples)

    @property
    def argument_examples(self):
        """:rtype: dict[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
        return self.span[0]

    @argument_examples.setter
    def argument_examples(self, candidates):
        """:type candidates: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
        # going to assume for now, that each role has max 1 argument
        candidate_dict = {arg.event_role.lower(): arg for arg in candidates}
        self.span = (candidate_dict, self.trigger_examples)

    def _allocate_arrays(self, hyper_params, embedding_vector_size,
                         _none_token_index, _features):
        """
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        float_type = 'float32'
        role_window = 2 * hyper_params.neighbor_distance + 1

        self.trigger_window_vector = None
        self.agent_window_vector = None
        self.patient_window_vector = None

        self.trigger_window_vector = self.trigger_example.window_data_vector

        if 'agent' in self.argument_examples:
            self.agent_window_vector = self.argument_examples['agent'].argument_window_vector
        else:
            self.agent_window_vector = np.zeros((role_window, embedding_vector_size), dtype=float_type)

        if 'patient' in self.argument_examples:
            self.patient_window_vector = self.argument_examples['patient'].argument_window_vector
        else:
            self.patient_window_vector = np.zeros((role_window, embedding_vector_size), dtype=float_type)

class EventFramePairExample(BinaryDatapoint):

    def __init__(self, eg1, eg2, label_string):
        """
        :param eg1: nlplingo.tasks.eventframe.example.EventFrameExample
        :param eg2: nlplingo.tasks.eventframe.example.EventFrameExample
        :param label_string: 'SAME' or 'DIFFERENT'
        """
        super(EventFramePairExample, self).__init__(
            eg1, eg2, None, None, label_string, None)
        self.label = pair_labels[label_string]

    @property
    def eg1(self):
        """:rtype: nlplingo.tasks.eventframe.example.EventFrameExample"""
        return self.left_datapoint

    @eg1.setter
    def eg1(self, frame_candidate):
        """:type frame_candidate: nlplingo.tasks.eventframe.example.EventFrameExample"""
        self.left_datapoint = frame_candidate

    @property
    def eg2(self):
        """:rtype: nlplingo.tasks.eventframe.example.EventFrameExample"""
        return self.right_datapoint

    @eg2.setter
    def eg2(self, frame_candidate):
        """:type frame_candidate: nlplingo.tasks.eventframe.example.EventFrameExample"""
        self.right_datapoint = frame_candidate

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
        :type features: nlplingo.event.eventframe.feature.EventFramePairFeature
        :rtype: dict[str:numpy.ndarray]
        """
        d = dict()

        d[features.c_trigger_window_vector1] = self.eg1.trigger_window_vector
        d[features.c_trigger_window_vector2] = self.eg2.trigger_window_vector

        d[features.c_agent_window_vector1] = self.eg1.agent_window_vector
        d[features.c_agent_window_vector2] = self.eg2.agent_window_vector

        d[features.c_patient_window_vector1] = self.eg1.patient_window_vector
        d[features.c_patient_window_vector2] = self.eg2.patient_window_vector

        d['label'] = self.label

        return d





