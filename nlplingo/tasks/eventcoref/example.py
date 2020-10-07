import numpy as np

from nlplingo.tasks.common.binary.binary_event_event import BinaryEventEvent

class EventCorefExample(BinaryEventEvent):
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
        super(EventCorefExample, self).__init__(arg0, arg1, event_domain, label_str)