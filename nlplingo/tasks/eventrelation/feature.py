
from nlplingo.tasks.common.binary.binary_event_event import BinaryEventEventFeatureGenerator
from nlplingo.common.utils import int_type
import numpy as np

class EventEventRelationFeatureGenerator(BinaryEventEventFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EventEventRelationFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :type example: BinaryEventEvent
        """
        super(EventEventRelationFeatureGenerator, self).populate_example(example)

        # set label
        # TODO: move this label code somewhere better, ideally with 1-hot toggle
        self.is_pytorch = 'engine' in self.extractor_params and (self.extractor_params['engine'] == 'pytorch')
        num_labels = len(example.event_domain.eer_types)
        if self.is_pytorch:
            example.label = np.zeros(1, dtype=int_type) # a single integer representing the class label
        else:
            example.label = np.zeros(num_labels, dtype=int_type) # a one-hot encoding of the label

        if example.eer_type is not None:
            if not self.is_pytorch:
                example.label[example.get_eer_type_index()] = 1
            else:
                example.label[0] = example.get_eer_type_index()

"""
class EventEventRelationFeatureGeneratorFactory(FeatureGeneratorFactory):
    def initialize_example_values(self, example, tokens):
        anchor1 = example.anchor1
         :type: nlplingo.text.text_span.Anchor 
        anchor2 = example.anchor2
         :type: nlplingo.text.text_span.Anchor 

        anchor1_token_index = anchor1.head().index_in_sentence
        anchor1_token_indices = Struct(start=anchor1_token_index,
                                       end=anchor1_token_index, head=anchor1_token_index)

        anchor2_token_index = anchor2.head().index_in_sentence
        anchor2_token_indices = Struct(start=anchor2_token_index,
                                       end=anchor2_token_index, head=anchor2_token_index)

        arg1_token_indices = anchor1_token_indices
        arg2_token_indices = anchor2_token_indices

        return locals()
"""