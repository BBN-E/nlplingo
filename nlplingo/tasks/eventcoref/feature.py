from nlplingo.tasks.common.binary.binary_event_event import BinaryEventEventFeatureGenerator

class EventCorefFeatureGenerator(BinaryEventEventFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EventCorefFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :type example: BinaryEventEvent
        """
        super(EventCorefFeatureGenerator, self).populate_example(example)