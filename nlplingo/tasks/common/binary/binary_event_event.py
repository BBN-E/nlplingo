from nlplingo.tasks.common.binary.binary_both_within_sentence import BinaryBothWithinSentence, BinaryBothWithinSentenceFeatureGenerator
from abc import ABC, abstractmethod

class BinaryEventEvent(BinaryBothWithinSentence):
    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type arg0: nlplingo.tasks.common.unary.within_sentence.EventWithinSentence
        :type arg1: nlplingo.tasks.common.unary.within_sentence.EventWithinSentence
        """
        super(BinaryEventEvent, self).__init__(arg0, arg1, event_domain, label_str)
        self.label = label_str # TODO: change to self.label_str?

class BinaryEventEventFeatureGenerator(BinaryBothWithinSentenceFeatureGenerator, ABC):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(BinaryEventEventFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    @abstractmethod
    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        super(BinaryEventEventFeatureGenerator, self).populate_example(example)