from nlplingo.tasks.common.binary.binary_entity_entity import BinaryEntityEntityFeatureGenerator

class EntityCorefFeatureGenerator(BinaryEntityEntityFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EntityCorefFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :type example: BinaryEntityEntity
        """
        super(EntityCorefFeatureGenerator, self).populate_example(example)