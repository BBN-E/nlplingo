from nlplingo.tasks.common.binary.binary_both_within_sentence import BinaryBothWithinSentence, BinaryBothWithinSentenceFeatureGenerator

class BinaryEntityEntity(BinaryBothWithinSentence):

    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type arg0: nlplingo.tasks.common.unary.within_sentence.EntityWithinSentence
        :type arg1: nlplingo.tasks.common.unary.within_sentence.EntityWithinSentence
        """
        super(BinaryEntityEntity, self).__init__(arg0, arg1, event_domain, label_str)
        self.label = label_str # TODO: change to self.label_str?

class BinaryEntityEntityFeatureGenerator(BinaryBothWithinSentenceFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(BinaryEntityEntityFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        super(BinaryEntityEntityFeatureGenerator, self).populate_example(example)
