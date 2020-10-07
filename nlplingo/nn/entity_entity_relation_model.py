from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
import logging

from keras.models import Model

from keras.callbacks import EarlyStopping

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel
from nlplingo.nn.cross_task_model import WithinSentence

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)

class WithinSentenceEntityRelationModel(WithinSentence):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        super(WithinSentenceEntityRelationModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.num_output = len(event_domain.entity_relation_types)
        self.event_domain_rel = event_domain.entity_relation_types
        self.create_model()

    @property
    def none_label_index(self):
        return self.event_domain.get_eer_type_index('NA')

class EntityRelationModel(ExtractionModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(EntityRelationModel, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)

        self.num_output = len(eer_domain.entity_relation_types)

    @property
    def none_label_index(self):
        return self.event_domain.get_entity_relation_type_index('None')


class EntityRelationKerasModel(EntityRelationModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls EntityRelationModel init (for task-specific model parameters)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(EntityRelationKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(EntityRelationKerasModel, self).create_model()


class MultiLayerEntityRelationModelEmbedded(EntityRelationKerasModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(MultiLayerEntityRelationModelEmbedded, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        super(MultiLayerEntityRelationModelEmbedded, self).create_model()
        model_input_dict = dict()
        outputs_to_merge = []

        self.layers.add_binary_window_layer(
            "arg0_arg1_window_vector", model_input_dict, outputs_to_merge,
            self.layers.EmbeddingLayer.NONE)

        hidden_output = self.layers.build_hidden_layers(outputs_to_merge[0])

        # historically self.activation = 'softmax'
        model_outputs = []
        self.layers.add_decision_layer([hidden_output], model_outputs, dropout=False)

        self.compile(model_outputs, model_input_dict)

