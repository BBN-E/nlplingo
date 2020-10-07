from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import logging
import os

import keras
import numpy as np
from keras.models import Model

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.model.base_model import KerasExtractionModel
from nlplingo.common.serialize_disk import load_class_weights

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


class TriggerModel(ExtractionModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.eventtrigger.feature.EventTriggerFeature
        """
        super(TriggerModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.number_of_entity_bio_types = len(event_domain.entity_bio_types)
        self.num_output = len(event_domain.event_types)
        # TODO remove or generalize this:
        # currently, this attribute is only used by TriggerModels and is set
        # only in KerasExtractionModels (and there are only TriggerKerasModels)
        self.is_binary = None

    @property
    def none_label_index(self):
        return self.event_domain.get_event_type_index('None')


class TriggerKerasModel(TriggerModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls TriggerModel init (for task-specific model params)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(TriggerKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(TriggerKerasModel, self).create_model()


class CNNTriggerModel(TriggerKerasModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.eventtrigger.feature.EventTriggerFeature
        """
        super(CNNTriggerModel, self).__init__(
            params,
            extractor_params,
            event_domain,
            embeddings,
            hyper_params,
            features)
        self.create_model()

    def create_model(self):
        super(CNNTriggerModel, self).create_model()
        model_input_dict = dict()
        outputs_to_merge_1 = []

        self.layers.add_sentence_word_embedding_layer(
            "sentence_word_embedding",
            model_input_dict,
            outputs_to_merge_1,
            self.layers.EmbeddingLayer.PRETRAINED)

        self.layers.add_sentence_word_embedding_layer(
            "sentence_word_embedding_vector",
            model_input_dict,
            outputs_to_merge_1,
            self.layers.EmbeddingLayer.NONE)

        # For each word the pos_array_input defines the distance to the target work.
        # Embed each distance into an 'embedding_vec_length' dimensional vector space
        self.layers.add_unary_position_layer(
            "unary_word_position", model_input_dict, outputs_to_merge_1)

        # Sentence feature input is the result of merging word vectors and embeddings
        self.layers.add_sentence_ner_embedding_layer(
            "sentence_ner_type", model_input_dict, outputs_to_merge_1,
            self.num_ne_bio_types)

        merged = self.layers.merge(outputs_to_merge_1)

        outputs_to_merge_2 = []
        self.layers.add_convolutional_layers(
            merged, outputs_to_merge_2, self.layers.BorderMode.SAME)

        self.layers.add_unary_window_layer(
            "unary_window", model_input_dict, outputs_to_merge_2,
            self.layers.EmbeddingLayer.PRETRAINED)

        self.layers.add_unary_window_layer(
            "unary_window_vector", model_input_dict, outputs_to_merge_2,
            self.layers.EmbeddingLayer.NONE)

        # Hierarchical Transfer (implemented as optional feature)
        weighted_layer = self.layers.apply_hierarchical_transfer_layer(
            outputs_to_merge_2)

        # historically self.activation = 'softmax' or 'sigmoid'
        model_outputs = []
        self.layers.add_decision_layer([weighted_layer], model_outputs)

        self.compile(model_outputs, model_input_dict)


class MultiLayerTriggerModelEmbedded(TriggerKerasModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.eventtrigger.feature.EventTriggerFeature
        """
        super(MultiLayerTriggerModelEmbedded, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        super(MultiLayerTriggerModelEmbedded, self).create_model()
        model_input_dict = dict()
        layer_list = []

        # word vectors for window around the unary datapoint
        self.layers.add_unary_window_layer(
            "unary_window_vector", model_input_dict, layer_list,
            self.layers.EmbeddingLayer.NONE)

        # word vectors for the entire sentence containing the unary datapoint
        self.layers.add_sentence_word_embedding_layer(
            "sentence_word_embedding_vector",
            model_input_dict,
            layer_list,
            self.layers.EmbeddingLayer.NONE)

        # For each word the pos_array_input defines distance to the target word.
        # Embed each distance into an 'embedding_vec_length'-D vector space
        self.layers.add_unary_position_layer(
            "unary_word_position", model_input_dict, layer_list)

        self.layers.add_sentence_ner_embedding_layer(
            "sentence_ner_type", layer_list, model_input_dict,
            self.num_ne_bio_types)

        # Hidden layer input is merged word vecs and sentence embeddings
        hidden_input = self.layers.merge(layer_list)
        hidden_output = self.layers.build_hidden_layers(hidden_input)

        # historically self.activation = 'softmax' or 'sigmoid'
        model_outputs = []
        self.layers.add_decision_layer([hidden_output], model_outputs)

        self.compile(model_outputs, model_input_dict)
