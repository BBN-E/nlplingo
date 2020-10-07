from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
import logging

from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model
from keras.constraints import maxnorm

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel

from nlplingo.nn.cross_task_model import WithinSentence

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)

class WithinSentenceArgumentModel(WithinSentence):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        super(WithinSentenceArgumentModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.num_output = len(event_domain.event_roles)
        self.event_domain_rel = event_domain.event_roles
        self.create_model()

    @property
    def none_label_index(self):
        return self.event_domain.get_event_role_index('None')

class ArgumentModel(ExtractionModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventargument.feature.EventArgumentFeature
        """
        super(ArgumentModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

        self.num_output = len(event_domain.event_roles)

    @property
    def none_label_index(self):
        return self.event_domain.get_event_role_index('None')


class ArgumentKerasModel(ArgumentModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls ArgumentModel init (prepares task-specific model parameters)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(ArgumentKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(ArgumentKerasModel, self).create_model()


class CNNArgumentModel(ArgumentKerasModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventargument.feature.EventArgumentFeature
        """
        super(CNNArgumentModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        super(CNNArgumentModel, self).create_model()
        model_input_dict = dict()
        outputs_to_merge1 = []

        self.layers.add_sentence_word_embedding_layer(
            'arg0_sentence_word_embedding',
            model_input_dict,
            outputs_to_merge1,
            self.layers.EmbeddingLayer.PRETRAINED)

        # # TODO this feature hasn't been implemented
        # self.layers.add_sentence_word_embedding_layer(
        #     "arg0_sentence_word_embedding_vector",
        #     model_input_dict,
        #     outputs_to_merge1,
        #     self.layers.EmbeddingLayer.NONE)

        self.layers.add_unary_position_layer(
            "arg0_word_position", model_input_dict, outputs_to_merge1)

        self.layers.add_unary_position_layer(
            "arg1_word_position", model_input_dict, outputs_to_merge1)

        self.layers.add_event_embedding_layer(
            "event_embeddings", model_input_dict, outputs_to_merge1, with_dropout=False)

        self.layers.add_sentence_ner_embedding_layer(
            "arg0_sentence_ner_type", model_input_dict, outputs_to_merge1)

        merged = self.layers.merge(outputs_to_merge1)

        outputs_to_merge2 = []
        self.layers.add_convolutional_layers(
            merged, outputs_to_merge2, self.layers.BorderMode.VALID)

        self.layers.add_binary_window_layer(
            "arg0_arg1_window_vector", model_input_dict, outputs_to_merge2,
            self.layers.EmbeddingLayer.NONE)

        self.layers.add_binary_window_layer(
            "arg0_arg1_window", model_input_dict, outputs_to_merge2,
            self.layers.EmbeddingLayer.PRETRAINED)

        self.layers.add_unary_window_layer(
            "arg1_window", model_input_dict, outputs_to_merge2,
            self.layers.EmbeddingLayer.PRETRAINED)

        # historically self.activation = 'softmax'
        model_outputs = []
        self.layers.add_decision_layer(outputs_to_merge2, model_outputs)

        self.compile(model_outputs, model_input_dict)
        print(self.model.summary())


class GNNArgumentModel(ArgumentModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        # PyTorch model
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.argument.feature.EventArgumentFeature
        """
        super(GNNArgumentModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params,
                                               features)
        self.hyper_params.dict['num_ne_types'] = self.num_ne_types
        self.create_model()

    def create_model(self):
        from gcn_prune_model.trainer import GCNTrainer
        self.model = GCNTrainer


class MultiLayerArgumentModelEmbedded(ArgumentKerasModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventargument.feature.EventArgumentFeature
        """
        super(MultiLayerArgumentModelEmbedded, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        super(MultiLayerArgumentModelEmbedded, self).create_model()
        model_input_dict = dict()
        outputs_to_merge = []

        self.layers.add_binary_window_layer(
            "arg0_arg1_window_vector", model_input_dict, outputs_to_merge,
            self.layers.EmbeddingLayer.NONE)

        self.layers.add_unary_window_layer(
            "arg1_window_vector", model_input_dict, outputs_to_merge,
            self.layers.EmbeddingLayer.NONE)

        hidden_output = self.layers.build_hidden_layers(outputs_to_merge[0])

        to_output_layer_list = [hidden_output]
        self.layers.add_event_embedding_layer(
            "event_embeddings", model_input_dict, to_output_layer_list, with_dropout=True)

        # historically self.activation = 'softmax'
        model_outputs = []
        self.layers.add_decision_layer(to_output_layer_list, model_outputs)

        self.compile(model_outputs, model_input_dict)

