from __future__ import absolute_import

from enum import Enum
import logging

from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import multiply
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding

from nlplingo.nn.keras_models.common.layers import WeightedHierarchyLayer
from nlplingo.nn.keras_models.common.losses import contrastive_loss
from nlplingo.nn.keras_models.common.losses import masked_bce


logger = logging.getLogger(__name__)


class LayerCreator(object):

    allowed_binary_activations = {u'sigmoid': u'sigmoid'}
    allowed_categorical_activations = {u'softmax': u'softmax'}
    allowed_activations = {
        k: v for k, v in list(allowed_binary_activations.items()) +
                         list(allowed_categorical_activations.items())}
    allowed_losses = {
        u'categorical_crossentropy': u'categorical_crossentropy',
        u'binary_crossentropy': u'binary_crossentropy',
        u'contrastive_loss': contrastive_loss,
        u'masked_bce': masked_bce,
    }

    class EmbeddingLayer(Enum):
        """
        This enumeration is used to determine what type of keras Embedding layer
        should be built for a new group of layers, if any.  An Embedding layer
        converts an int representing a 1-hot vector into a dense vector of a
        specified length.  Members of this class include:
        - NONE: use no Embedding.  The input is already dense.
        - RANDOMLY_INITIALIZED: create a new, trainable embedding.
        - PRETRAINED: use pretrained word embeddings to set the weights and
            shape of an embedding.  The same Embedding layer is used for all
            features/layers that use word embeddings.  Trainability is
            determined by a hyperparameter.
        """
        NONE = "NONE"
        PRETRAINED = "PRETRAINED"
        RANDOMLY_INITIALIZED = "RANDOMLY_INITIALIZED"

    class BorderMode(Enum):
        SAME = u'same'
        VALID = u'valid'

    def __init__(self,
                 activated_features,
                 extractor_parameters,
                 hyper_parameters,
                 event_domain,
                 output_dimensions,
                 pretrained_embeddings,
                 ):
        # attributes from extractor parameters dict
        self.word_embedding_dims = extractor_parameters.get(
            'embeddings', {'vector_size': None})['vector_size']
        self.decision_function = LayerCreator.allowed_activations[
            extractor_parameters['activation_function']]
        self.is_binary = self.decision_function in LayerCreator.allowed_binary_activations.values()
        if self.is_binary:
            logger.info("Model type has been identified as a binary classifier.")
        else:
            logger.info("Model type has been identified as an argmax classifier.")
        self.loss = LayerCreator.allowed_losses[extractor_parameters['loss_function']]
        self.is_hierarchical = extractor_parameters.get('hierarchical', False)

        # attributes from hyperparameters
        self.hyper_parameters = hyper_parameters
        for attribute in ['max_sentence_length',  # maximum sentence length
                          'neighbor_distance',  # size of half of window
                          'hidden_layers',  # sizes of hidden layers
                          'train_embeddings',  # whether word embeddings are trainable
                          'position_embedding_vector_length',  # position_embedding_dims
                          'entity_embedding_vector_length',  # entity_embedding_dims
                          'dropout',  # dropout_rate
                          'number_of_feature_maps',  # number_of_convolution_maps
                          'cnn_filter_lengths',  # sizes_of_convolution_maps
                          ]:
            setattr(self, attribute, getattr(hyper_parameters, attribute, None))

        # attributes from event_domain
        self.domain = event_domain
        self.number_of_entity_types = len(event_domain.entity_types)
        self.number_of_event_types = len(event_domain.event_types)

        self.activated_features = activated_features
        self.pretrained_embeddings = pretrained_embeddings

        # attributes assigned outside of ExtractionModel
        self.output_dimensions = output_dimensions

        self._common_embedding_layer = None

    @property
    def common_embedding_layer(self):
        if self._common_embedding_layer is None:  # no common embedding yet
            self._common_embedding_layer = (
                self.new_pretrained_embedding_layer(u'shared_word'))
        return self._common_embedding_layer

    @staticmethod
    def input_layer(feature_name, shape):
        """
        :param feature_name: name of activated feature attribute
        :param shape: shape and dimensionality of input layer
        """
        input_name = u'{}_input'.format(feature_name)
        if len(shape) > 1:
            return Input(shape=shape, name=input_name, dtype=u'float32')
        else:
            return Input(shape=shape, name=input_name, dtype=u'int32')

    @staticmethod
    def flat(layer):
        return Flatten()(layer)

    def drop_out(self, layer):
        return Dropout(self.dropout)(layer)

    def dense(self, layer, size, activation='relu', constraint=None, name=None):
        densely_connected = self._get_dense(size, activation, constraint, name)
        return densely_connected(layer)

    @staticmethod
    def merge(layers):
        if len(layers) == 1:
            return layers[0]
        elif len(layers) > 1:
            return concatenate(layers, axis=-1)
        else:  # could raise error, but the issue will arise immediately anyway
            return None

    @staticmethod
    def new_embedding_layer(shape, feature_name, weights=None, trainable=True,
                            init='uniform'):
        name = u'{}_embedding'.format(feature_name)
        weights = [weights] if weights is not None else weights
        return Embedding(
            shape[0], shape[1], weights=weights, trainable=trainable,
            embeddings_initializer=init, name=name)

    def new_pretrained_embedding_layer(self, feature_name):
        embedding_layer = self.new_embedding_layer(
            self.pretrained_embeddings.shape, feature_name,
            self.pretrained_embeddings, self.train_embeddings,
            'glorot_uniform')
        return embedding_layer

    def add_sentence_word_embedding_layer(
            self, feature_name, inputs_dict, output_list, embeddings):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        :param embeddings: Enum: use common, new, or no Embeddings layer
        """
        if feature_name in self.activated_features:
            if embeddings == self.EmbeddingLayer.PRETRAINED:
                shape = None
            elif embeddings == self.EmbeddingLayer.NONE:
                shape = None
            else:
                # RANDOMLY_INITIALIZED is probably not what you want to use
                # here, and if you do it should probably use a dedicated
                # embedding shape determined in hyperparameters
                raise NotImplementedError(
                    "Invalid embeddings selected for sentence word embedding "
                    "layer: {}".format(embeddings.name))

            in_layer, out_layer = self._make_sentence_layers(
                feature_name, shape, embeddings)
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_unary_position_layer(self, feature_name, inputs_dict, output_list):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        """
        if feature_name in self.activated_features:
            shape = (2 * self.max_sentence_length, self.position_embedding_vector_length)
            in_layer, out_layer = self._make_sentence_layers(
                feature_name, shape, self.EmbeddingLayer.RANDOMLY_INITIALIZED)
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_unary_window_layer(self, feature_name, inputs_dict, output_list,
                               embeddings):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        :param embeddings: Enum: use common, new, or no Embeddings layer
        """
        if feature_name in self.activated_features:
            in_layer, out_layer = self._make_window_layers(feature_name,
                                                           embeddings, 1)
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_binary_window_layer(self, feature_name, inputs_dict, output_list,
                                embeddings):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        :param embeddings: Enum: use common, new, or no Embeddings layer
        """
        if feature_name in self.activated_features:
            in_layer, out_layer = self._make_window_layers(feature_name,
                                                           embeddings, 2)
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_event_embedding_layer(self, feature_name, inputs_dict, output_list, with_dropout=False):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        :param with_dropout: whether or not to apply Dropout and Flatten
        """
        if feature_name in self.activated_features:
            # TODO this should have its own dimensionality (like entities)
            shape = (self.number_of_event_types, self.position_embedding_vector_length)
            in_layer, embedded_layer = self._make_sentence_layers(
                feature_name, shape, self.EmbeddingLayer.RANDOMLY_INITIALIZED)
            if with_dropout:
                drop_layer = self.drop_out(embedded_layer)
                out_layer = self.flat(drop_layer)
            else:
                out_layer = embedded_layer
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_sentence_ner_embedding_layer(
            self, feature_name, inputs_dict, output_list, number_of_entity_types=None):
        """
        :param feature_name: name of activated feature attribute
        :param inputs_dict: dict {feature_name: Input layer}
        :param output_list: list of Layers to which output will be appended
        :param number_of_entity_types: int override for number of entity types,
            for example if the feature uses BIO labels instead of plain labels
        """
        if feature_name in self.activated_features:

            if number_of_entity_types is None:
                number_of_entity_types = self.number_of_entity_types
            shape = (number_of_entity_types, self.entity_embedding_vector_length)

            # TODO determine equivalence:
            # originally this had Input(shape=(dims,1)) which is distinct from
            # Input(shape=(dims,)). Is that distinction meaningful to the model?

            in_layer, out_layer = self._make_sentence_layers(
                feature_name, shape, self.EmbeddingLayer.RANDOMLY_INITIALIZED)
            inputs_dict[feature_name] = in_layer
            output_list.append(out_layer)

    def add_convolutional_layers(self, input_layer, output_list, border_mode=None):
        """
        :param input_layer: a Layer used as input to convolution
        :param output_list: list of Layers to which outputs will be appended
        :param border_mode: Enum: convolution border mode
        """
        if border_mode is None:
            border_mode = LayerCreator.BorderMode.SAME

        max_pools = []
        for filter_size in self.cnn_filter_lengths:
            # Note: border_mode='same' to keep output the same width as input
            convolution = Convolution1D(
                self.number_of_feature_maps,
                filter_size,
                border_mode=border_mode.value,
                activation='relu')
            convolution_layer = convolution(input_layer)
            pooled_layer = GlobalMaxPooling1D()(convolution_layer)
            max_pools.append(pooled_layer)
        output_list.extend(max_pools)

    def add_decision_layer(self, input_layers, output_list, dropout=True):
        """
        :param input_layers: a list of Layers used as input to decision layer.
        :param output_list: list of Layers to which output will be appended
        :param dropout: whether to apply dropout to the decision layer
        """
        merged_input = self.merge(input_layers)
        if dropout:
            input_with_dropout = self.drop_out(merged_input)
        else:
            input_with_dropout = merged_input
        decision_layer = self.dense(input_with_dropout,
                                    self.output_dimensions,
                                    activation=self.decision_function,
                                    name="decision_layer")
        output_list.append(decision_layer)

    def build_hidden_layers(self, input_layer):
        """
        This method applies dropout to an MLP based on one input.
        :param input_layer: a Layers used as input to hidden layers.
        """
        # extractor config might not specify zero hidden layers, so default to 0
        hidden_layers = getattr(self, "hidden_layers", 0)
        for i, hidden_layer_size in enumerate(hidden_layers):
            name = u'hidden_dense_{}'.format(i)
            input_layer = self.dense(input_layer, hidden_layer_size, name=name)
        output_layer = self.drop_out(input_layer)
        return output_layer

    def build_shared_hidden_layers(self, input_layers):
        """
        This method applies multiplication to an MLP based on multiple inputs.
        Each hidden layer's weights are shared across all inputs.
        :param input_layers: a list of Layers used as input to hidden layers.
        """
        for i, hidden_layer_size in enumerate(self.hidden_layers):
            name = u'hidden_dense_{}'.format(i)
            densely_connected = self._get_dense(hidden_layer_size, name=name)
            for j in range(len(input_layers)):
                input_layers[j] = densely_connected(input_layers[j])
        multiplied_layer = multiply(input_layers)
        return multiplied_layer

    def apply_hierarchical_transfer_layer(self, input_layers):
        """
        This method applies weighting according to a learned hierarchical
        transfer learning layer, based on some inputs.  If the transfer learning
        feature is not turned on, simply merges (concatenates) them.

        Note: this is not implemented as a FeatureSetting.activated_features
        because there is nothing to assign to a given Datapoint.

        :param input_layers: a list of Layers used as input to weighting layer.
        """
        merged_layer = self.merge(input_layers)
        if self.is_hierarchical:
            # hierarchical transfer learning requires user to supply a hierarchy
            hierarchical_prior = self.domain.hierarchical_prior
            assert hierarchical_prior.shape == (self.output_dimensions,
                                                self.output_dimensions)
            weighting_layer = WeightedHierarchyLayer(
                out_dim=self.output_dimensions,
                prior=hierarchical_prior,
                previous_dim=merged_layer._keras_shape[-1]
            )
            transfer_layer = weighting_layer(merged_layer)
        else:
            transfer_layer = merged_layer
        return transfer_layer

    @staticmethod
    def _get_dense(size, activation='relu', constraint=None, name=None):
        if constraint is None:
            constraint = maxnorm(3)
        if name is None:
            densely_connected = Dense(size, activation=activation,
                                      kernel_constraint=constraint)
        else:
            densely_connected = Dense(size, activation=activation, name=name,
                                      kernel_constraint=constraint)
        return densely_connected

    def _make_window_layers(self, feature_name, embeddings, windows):
        """
        :param feature_name: name of activated feature attribute
        :param embeddings: Enum: use common, new, or no Embeddings layer
        :param windows: how many windows to use for input
        """
        # set up window input layer
        window_size = 2 * self.neighbor_distance + 1
        if embeddings == self.EmbeddingLayer.NONE:  # pre-embedded
            in_shape = (window_size * windows, self.word_embedding_dims)
            in_layer = self.input_layer(feature_name, in_shape)
        else:
            in_shape = (window_size * windows,)
            in_layer = self.input_layer(feature_name, in_shape)

        # set up window embedding layer
        if embeddings == self.EmbeddingLayer.NONE:
            # pre-embedded
            embedded_layer = in_layer
        elif embeddings == self.EmbeddingLayer.PRETRAINED:
            # use shared Embedding
            embedding_layer = self.common_embedding_layer
            embedded_layer = embedding_layer(in_layer)
        else:
            # RANDOMLY_INITIALIZED is probably not what you want to use here,
            # and if you do it should probably use a dedicated embedding shape
            # determined in hyperparameters
            raise IOError("Invalid embedding selection for window(s): {}"
                          .format(embeddings))

        out_layer = self.flat(embedded_layer)
        return in_layer, out_layer

    def _make_sentence_layers(self, feature_name, embedding_shape, embeddings):

        if embeddings == self.EmbeddingLayer.NONE:  # pre-embedded
            in_shape = (self.max_sentence_length, self.embeddings_vector_size)
            in_layer = self.input_layer(feature_name, in_shape)
            out_layer = in_layer

        else:
            in_shape = (self.max_sentence_length,)
            in_layer = self.input_layer(feature_name, in_shape)
            # use shared Embedding space
            if embeddings == self.EmbeddingLayer.PRETRAINED:
                embedding_layer = self.common_embedding_layer
            elif embeddings == self.EmbeddingLayer.RANDOMLY_INITIALIZED:
                # make single-use Embedding for this feature
                embedding_layer = self.new_embedding_layer(
                    embedding_shape, feature_name)
            else:
                raise NotImplementedError("Invalid sentence embedding selection"
                                          ": {}".format(embeddings.name))
            out_layer = embedding_layer(in_layer)

        return in_layer, out_layer

