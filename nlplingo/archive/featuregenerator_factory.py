
import numpy as np

from nlplingo.common.utils import int_type, float_type
from nlplingo.tasks.common.feature_meta import get_parameters, create_arg_store, caller_name
from nlplingo.tasks.common.feature_utils import _get_token_windows, _calculate_position_data, _calculate_position_data_variable

class FeatureGeneratorFactory(object):
    def __init__(self, extractor_params, features, usable_features, hyper_params):
        """
        Initialize various variables relevant to feature generation process.
        :param extractor_params: dict
        :param features: nlplingo.tasks.common.featuregenerator_actual.ActivatedFeatures
        :param usable_features: dict
        """
        if 'embeddings' in extractor_params:
            if 'none_token_index' in extractor_params['embeddings']:
                self.none_token_index = extractor_params['embeddings']['none_token_index']
            else:
                self.none_token_index = 1

            self.embedding_vector_size = extractor_params['embeddings']['vector_size']
        else:
            pass

        self.features = features
        self.usable_features = usable_features
        self.hyper_params = hyper_params
        self.example_value_strings = set() # activated example-specific values to compute
        self.target_attributes = dict()

        # collect example value strings that correspond to activated features
        # collect target attributes
        for usable_feature in usable_features:
            if getattr(self.features, usable_feature):
                gen_fn_name = usable_features[usable_feature]['gen_fn']
                self.example_value_strings.update(get_parameters(getattr(self, gen_fn_name)))
                init_fn_name = gen_fn_name.replace('assign_', 'init_')
                self.target_attributes.update({init_fn_name : usable_features[usable_feature]['attribute']})
                self.target_attributes.update({gen_fn_name : usable_features[usable_feature]['attribute']})

        # print(self.example_value_strings)

        # custom default values for hyperparameter attributes
        if not hasattr(hyper_params, 'use_position_padding'):
            self.hyper_params.use_position_padding = False

    # an abstract method
    def initialize_example_values(self, example, tokens, hyper_params):
        """
        Instantiate nlplingo example-specific information for feature generation process.
        :param example: nlplingo example
        :param tokens:
        :param hyper_params:
        :return:
        """
        raise NotImplementedError()

    def create_example_attribute(self, example, value, stack_frames=2):
        """
        Used in the initializer functions for feature generation.
        Sets the nlplingo example target attribute to a given value.
        Notably, the target attribute must be a reference: usually it is a numpy array.
        This function does not work well for attributes that have ill-defined shape.
        :param example:
        :param value:
        :param stack_frames: default stack_frames to look over
        :return:
        """
        attribute_name = self.target_attributes[caller_name(stack_frames=stack_frames)] # Be careful about using caller_name(). It only makes sense within this class's context
        setattr(example, attribute_name, value)
        return getattr(example, attribute_name)

    def generate_example(self, example, tokens):
        """
        Generate the activated features for a given nlplingo example.
        :param example: nlplingo example
        :param tokens: example tokens
        :param hyper_params: hyper_params
        :return:
        """
        example_values = self.initialize_example_values(example, tokens)
        arg_store = create_arg_store(self.example_value_strings, example_values)
        for usable_feature in self.usable_features:
            if getattr(self.features, usable_feature):
                gen_fn_name = self.usable_features[usable_feature]['gen_fn']
                feature_gen = getattr(self, gen_fn_name + '_arg_store')
                feature_gen(arg_store)

    # window feature
    def init_lexical_data(self, example, neighbor_distance):
        """
        :param example: nlplingo example
        :param neighbor_distance: window distance around each argument
        """
        arg1_window = 2 * neighbor_distance + 1
        arg2_window = 2 * neighbor_distance + 1
        lex_data = np.zeros(arg1_window + arg2_window, dtype=int_type)
        lex_data[:] = self.none_token_index
        return self.create_example_attribute(example, lex_data)

    def assign_lexical_data(self, example,
                            arg1_token_indices, arg2_token_indices, tokens):
        """
        +1
        :type example: nlplingo example
        :type lexical_data_attribute: str
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        """

        neighbor_distance = self.hyper_params.neighbor_distance
        target = self.init_lexical_data(example, neighbor_distance)

        # get the local token windows around the arg1 and arg2
        token_windows = _get_token_windows(tokens, neighbor_distance, arg1_token_indices, arg2_token_indices)
        for (i, token) in token_windows:
            target[i] = token.vector_index  # local window around arg1 and arg2

    # sentence embeddings
    def init_sentence_vector_data(self, example):
        sentence_data_vector = np.zeros((self.hyper_params.max_sentence_length, self.embedding_vector_size), dtype=float_type)
        return self.create_example_attribute(example, sentence_data_vector)

    def assign_sentence_vector_data(self, tokens, example):
        """
        +1
        Capture the actual word embeddings, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        target = self.init_sentence_vector_data(example)
        for i, token in enumerate(tokens):
            if i >= self.hyper_params.max_sentence_length:
                break
            if token.word_vector is not None:
                target[i, :] = token.word_vector

    # decode triplets (for eer relations, OpenNRE)
    def assign_decode_triplet(self, example):
        """
        +1
        Capture the raw untokenized relation (dict containing sentence along with offsets)
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        self.create_example_attribute(example, example.to_triplet())

    # position feature
    def init_position_data(self, example, max_sentence_length):
        pos_data = np.zeros(max_sentence_length, dtype=int_type)
        return self.create_example_attribute(example, pos_data)

    def assign_position_data(self, example,
                             arg1_token_indices):
        """
        +1
        :type arg1_token_indices: nlplingo.common.utils.Struct
        """
        max_sentence_length = self.hyper_params.max_sentence_length
        target = self.init_position_data(example, max_sentence_length)
        target[:] = _calculate_position_data(arg1_token_indices, max_sentence_length)

    # position feature, depending on variable sentence length
    # this function needs to be used within another function, not by itself
    def create_position_data_variable(self, example, tokens,
                                      token_indices):
        """
        if not padded , assigned attribute has shape np.array(length of tokens)
               padded , assigned attribute has shape np.array(max_sentence_length)
        +1
        :type token_indices: nlplingo.common.utils.Struct
        """
        max_sentence_length = self.hyper_params.max_sentence_length
        padded = self.hyper_params.use_position_padding
        result = _calculate_position_data_variable(tokens, token_indices, max_sentence_length, padded=padded)
        self.create_example_attribute(example, np.asarray(result), stack_frames=3)

    # position feature, depending on variable sentence length
    def assign_position_data_variable_arg1(self, example, tokens,
                             arg1_token_indices):
        """
        Assign arg1 position tokens.
        :type arg1_token_indices: nlplingo.common.utils.Struct
        """
        self.create_position_data_variable(example, tokens, arg1_token_indices)

    # position feature, depending on variable sentence length
    def assign_position_data_variable_arg2(self, example, tokens,
                             arg2_token_indices):
        """
        Create arg2 position tokens.
        :type arg2_token_indices: nlplingo.common.utils.Struct
        """
        self.create_position_data_variable(example, tokens, arg2_token_indices)

    # window embeddings
    def init_window_vector_data(self, example, neighbor_distance):
        """
        :param example: nlplingo.tasks.common.example.Example
        :param neighbor_distance: window distance around each argument
        """
        arg1_window = 2 * neighbor_distance + 1
        arg2_window = 2 * neighbor_distance + 1
        embedding_vector_size = self.embedding_vector_size
        window_data_vector = np.zeros((arg1_window + arg2_window, embedding_vector_size), dtype=float_type)
        return self.create_example_attribute(example, window_data_vector)

    def assign_window_vector_data(self, example,
                                  arg1_token_indices, arg2_token_indices, tokens):
        """We want to capture [word-on-left , target-word , word-on-right]
        :type example: nlplingo.tasks.common.example.Datapoint
        :type window_vector_data_attribute: str
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        """
        neighbor_distance = self.hyper_params.neighbor_distance
        target = self.init_window_vector_data(example, neighbor_distance)

        # get the local token windows around the arg1 and arg2
        token_windows = _get_token_windows(tokens, neighbor_distance, arg1_token_indices, arg2_token_indices)
        for (i, token) in token_windows:
            if token.word_vector is not None:
                target[i, :] = token.word_vector

