from nlplingo.tasks.common.binary.binary_both_within_sentence import BinaryBothWithinSentence, BinaryBothWithinSentenceFeatureGenerator

class BinaryEventEntity(BinaryBothWithinSentence):

    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type arg0: nlplingo.tasks.common.unary.within_sentence.EventWithinSentence
        :type arg1: nlplingo.tasks.common.unary.within_sentence.EntityWithinSentence
        """
        super(BinaryEventEntity, self).__init__(arg0, arg1, event_domain, label_str)

    def event_embeddings(self, max_length):
        """
        For use as a feature.
        :param max_length:
        :return:
        """
        return self.arg0.event_embeddings(max_length)

class BinaryEventEntityFeatureGenerator(BinaryBothWithinSentenceFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(BinaryEventEntityFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        super(BinaryEventEntityFeatureGenerator, self).populate_example(example)
        if hasattr(self.hyper_params, 'max_sentence_length'):
            self.assign_example(example, "event_embeddings", [self.hyper_params.max_sentence_length])