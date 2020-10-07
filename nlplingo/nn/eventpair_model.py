from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import logging
import os

import keras
import numpy as np
from keras.layers import Input
from keras.layers.embeddings import Embedding
#from keras.layers import Subtract
from keras.layers import Multiply
from keras.layers import Lambda
from keras.models import Model

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


from keras import backend as K

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


class EventPairModel(ExtractionModel):
    # +1
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventpair.feature.EventPairFeature
        """
        super(EventPairModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.num_output = 1

    @property
    def none_label_index(self):
        return 0


class EventPairKerasModel(EventPairModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls EventPairModel init (for task-specific model params)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(EventPairKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(EventPairKerasModel, self).create_model()


class EventPairModelEmbeddedTrigger(EventPairKerasModel):
    # +1
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventpair.feature.EventPairFeature
        """
        super(EventPairModelEmbeddedTrigger, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def pretrained_trigger_model(self, trigger_extractor, is_trainable=False):
        trigger_model = trigger_extractor.extraction_model.model

        print("Original trigger model:")
        print(trigger_model.summary())

        # Get rid of the Dense layer and freeze/unfreeze layers
        trigger_model.layers.pop()
        for layer in trigger_model.layers:
            layer.trainable = is_trainable

        print("Current trigger model:")
        print(trigger_model.summary())

        self.trigger_model = trigger_model

        self.trigger_model2 = keras.models.clone_model(trigger_model)
        self.trigger_model2.set_weights(trigger_model.get_weights())
        self.trigger_model2.layers.pop()
        print("Copy trigger model:")
        print(self.trigger_model2.summary())

    # +1
    def create_model(self):
        super(EventPairModelEmbeddedTrigger, self).create_model()
        # TODO this model has some quirks to dismantle like this:
        self.layers.output_dimensions = 1
        model_input_dict = dict()

        trigger_model1 = self.trigger_model
        input_window1 = trigger_model1.inputs[0]
        model_input_dict[self.features.c_trigger_window_vector1] = input_window1
        output_features1 = trigger_model1.get_layer(u'hidden_dense_1').output

        trigger_model2 = self.trigger_model2
        for layer in trigger_model2.layers:
            layer.name = layer.name + str("_2")
        input_window2 = trigger_model2.inputs[0]
        model_input_dict[self.features.c_trigger_window_vector2] = input_window2
        output_features2 = trigger_model2.get_layer(u'hidden_dense_1_2').output

        lex1, lex2 = output_features1, output_features2
        lex_layers = [lex1, lex2]
        x5 = self.layers.build_shared_hidden_layers(lex_layers)

        # historically self.activation = 'sigmoid'
        model_outputs = []
        self.layers.add_decision_layer([x5], model_outputs)

        self.compile(model_outputs, model_input_dict, metrics=['accuracy'])

        print("Eventpair model:")
        print(self.model.summary())
