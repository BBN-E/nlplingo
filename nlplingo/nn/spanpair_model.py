from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import logging

import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K

import tensorflow as tf

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


# currently not used
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.mean(x * y, axis=-1, keepdims=True)

# currently not used
def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


class SpanPairModel(ExtractionModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventpair.feature.EventPairFeature
        """
        super(SpanPairModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.num_output = 1

    @property
    def none_label_index(self):
        return 0


class SpanPairKerasModel(SpanPairModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls SpanPairModel init (for task-specific model params)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(SpanPairKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(SpanPairKerasModel, self).create_model()


class SpanPairModelEmbedded(SpanPairKerasModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventpair.feature.EventPairFeature
        """
        super(SpanPairModelEmbedded, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    # ref: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    def f1_loss(self, y_true, y_pred):
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)

    def create_model(self):
        super(SpanPairModelEmbedded, self).create_model()
        # TODO this model has some quirks to dismantle like this:
        self.layers.output_dimensions = 1
        model_input_dict = dict()
        lex_layers = []

        self.layers.add_unary_window_layer(
            "arg0_window_vector", model_input_dict, lex_layers,
            self.layers.EmbeddingLayer.NONE)
        self.layers.add_unary_window_layer(
            "arg1_window_vector", model_input_dict, lex_layers,
            self.layers.EmbeddingLayer.NONE)

        x5 = self.layers.build_shared_hidden_layers(lex_layers)

        # historically self.activation = 'sigmoid'
        model_outputs = []
        self.layers.add_decision_layer([x5], model_outputs)

        self.compile(model_outputs, model_input_dict, metrics=['accuracy'])
