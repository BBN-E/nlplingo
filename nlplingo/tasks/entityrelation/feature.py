from nlplingo.tasks.common.binary.binary_entity_entity import BinaryEntityEntityFeatureGenerator

class EntityRelationFeatureGenerator(BinaryEntityEntityFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EntityRelationFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :type example: BinaryEntityEntity
        """
        super(EntityRelationFeatureGenerator, self).populate_example(example)
        example.label[example.get_relation_type_index()] = 1
