from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import logging
import os

import numpy as np
import keras

from keras.callbacks import EarlyStopping

from keras.layers import Input
from keras.layers.embeddings import Embedding
#from keras.layers import Subtract
from keras.layers import Multiply
from keras.layers import Lambda
from keras.models import Model

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


from keras import backend as K


class EventFramePairModel(ExtractionModel):
    # +1
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventframe.feature.EventFramePairFeature
        """
        super(EventFramePairModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.num_output = 1

    @property
    def none_label_index(self):
        return 0


class EventFramePairKerasModel(EventFramePairModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls EventFramePairModel init (for task-specific model params)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(EventFramePairKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(EventFramePairKerasModel, self).create_model()


class EventFramePairModelEmbedded(EventFramePairKerasModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventpair.feature.EventPairFeature
        """
        super(EventFramePairModelEmbedded, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    # +1
    def create_model(self):
        super(EventFramePairModelEmbedded, self).create_model()
        # TODO this model has some quirks to dismantle like this:
        self.layers.output_dimensions = 1
        model_input_dict = dict()

        window_size = 2 * self.hyper_params.neighbor_distance + 1

        #### event frame 1
        trigger_window_input1 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32', name=u'trigger_window1')
        model_input_dict[self.features.c_trigger_window_vector1] = trigger_window_input1
        trigger1 = self.layers.flat(trigger_window_input1)

        agent_window_input1 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32',
                                      name=u'agent_window1')
        model_input_dict[self.features.c_agent_window_vector1] = agent_window_input1
        agent1 = self.layers.flat(agent_window_input1)

        patient_window_input1 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32',
                                    name=u'patient_window1')
        model_input_dict[self.features.c_patient_window_vector1] = patient_window_input1
        patient1 = self.layers.flat(patient_window_input1)

        lex1_components = [trigger_window_input1, agent_window_input1, patient_window_input1]
        lex1 = self.layers.flat(self.layers.merge(lex1_components, axis=-1))

        #### event frame 2
        trigger_window_input2 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32', name=u'trigger_window2')
        model_input_dict[self.features.c_trigger_window_vector2] = trigger_window_input2
        trigger2 = self.layers.flat(trigger_window_input2)

        agent_window_input2 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32',
                                    name=u'agent_window2')
        model_input_dict[self.features.c_agent_window_vector2] = agent_window_input2
        agent2 = self.layers.flat(agent_window_input2)

        patient_window_input2 = Input(shape=(window_size, self.embeddings_vector_size), dtype=u'float32',
                                      name=u'patient_window2')
        model_input_dict[self.features.c_patient_window_vector2] = patient_window_input2
        patient2 = self.layers.flat(patient_window_input2)

        lex2_components = [trigger_window_input2, agent_window_input2, patient_window_input2]
        lex2 = self.layers.flat(self.layers.merge(lex2_components, axis=-1))

        lex_layers = [lex1, lex2]
        x5 = self.layers.build_shared_hidden_layers(lex_layers)

        #conc = Concatenate(axis=-1)([x5, x4, x3])

        # historically self.activation = 'sigmoid'
        model_outputs = []
        self.layers.add_decision_layer([x5], model_outputs)

        self.compile(model_outputs, model_input_dict, metrics=['accuracy'])

