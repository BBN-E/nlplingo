from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import abc
import json
import logging
import numpy as np
import os

import keras
from keras.optimizers import Adadelta, SGD, RMSprop, Adam

from nlplingo.nn.constants import supported_pytorch_models
from nlplingo.nn.keras_models.common import keras_custom_objects

import time
from datetime import datetime
from shutil import copyfile

import random
import math
from nlplingo.nn.framework.sentence_re import SentenceRETrain

logger = logging.getLogger(__name__)


class ExtractionModel(abc.ABC):
    verbosity = 0

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.tasks.event_domain.EventDomain
        :type embeddings: dict[str : nlplingo.embeddings.word_embeddings.WordEmbedding]
        :type model_name: str
        :type features: object containing a 'feature_strings' attribute
        """
        self.hyper_params = hyper_params
        self.params = params
        self.extractor_params = extractor_params
        self.event_domain = event_domain
        self.num_event_types = len(event_domain.event_types)
        self.num_role_types = len(event_domain.event_roles)
        self.num_ne_types = len(event_domain.entity_types)
        self.num_ne_bio_types = None
        self.num_entity_relation_types = len(event_domain.entity_relation_types)
        self.num_eer_types = len(event_domain.eer_types)

        self.word_vec_length = 1                    # because we use word vector index

        self.embeddings_vector_size = None
        if 'embeddings' in extractor_params:
            self.embeddings_vector_size = extractor_params['embeddings']['vector_size']

        self.word_embeddings = None
        if embeddings is not None and 'word_embeddings' in embeddings:
            self.word_embeddings = embeddings['word_embeddings'].word_vec
            """:type: numpy.ndarray"""

        self.model_type = extractor_params['model_type']
        self.optimizer = self._configure_optimizer(extractor_params)
        self.model_file = extractor_params['model_file']

        self.data_keys = []
        self.num_output = None
        self.model_dir = None
        self.model = None

        self.id2label = dict([(v, k) for k, v in self.event_domain.event_roles.items()])

        self.trained_model = None

        self.features = features

        if 'engine' in extractor_params and (extractor_params['engine'] == 'pytorch'):
            import torch
            import random

            torch.manual_seed(extractor_params['seed'])
            np.random.seed(extractor_params['seed'])
            random.seed(1234)
            self.extractor_params['cuda'] = torch.cuda.is_available()

            if extractor_params.get('cpu', False):
                self.extractor_params['cuda'] = False
            elif extractor_params.get('cuda', False):
                torch.cuda.manual_seed(extractor_params['seed'])

        self.layers = None

    def _get_framework_class(self):
        if self.model_type in supported_pytorch_models:
            return SentenceRETrain
        else:
            raise Exception('model type ' + self.model_type + ' is not supported')

    def fit_txt(self, train_path, dev_path, test_path):
        # uses framework (with distinct initialization args)
        framework_class = self._get_framework_class()
        framework = framework_class(self.model, train_path, dev_path, test_path, self.extractor_params, self.hyper_params, self.features, self.event_domain)
        framework.train_model()

    def fit_model(self, train_data_list, train_label, test_data_list, test_label):
        # uses framework
        if self.extractor_params.get('engine') == 'pytorch':
            framework_class = self._get_framework_class()
            framework = framework_class(self.model, train_data_list, train_label, test_data_list, test_label, self.extractor_params, self.hyper_params, self.features, self.event_domain)
            framework.train_model()
        elif 'engine' not in self.extractor_params or (('engine' in self.extractor_params) and (self.extractor_params['engine'] == 'keras')):
            raise IOError(
                "Extractor engine in {'keras', None} but KerasExtractionModel "
                "should have implemented its own fit method overriding "
                "ExtractionModel.fit_model.  This error should no longer exist "
                "once KerasExtractionModel is part of framework_class system.")
        else:
            raise Exception('Only Keras or PyTorch engines are supported.')

    def _configure_optimizer(self, params):
        optimizer_params = params.get('optimizer', dict())
        tunable_params = {}
        if 'engine' not in self.extractor_params or (('engine' in self.extractor_params) and (self.extractor_params['engine'] == 'keras')):
            if optimizer_params.get('name') == 'SGD':
                tunable_params = {
                    'name': 'SGD',
                    'lr': optimizer_params.get('lr', 0.01),
                    'momentum': optimizer_params.get('momentum', 0.0),
                    'decay': optimizer_params.get('decay', 0.0),
                    'nesterov': optimizer_params.get('nesterov', False)
                }
                optimizer = SGD(
                    lr=tunable_params['lr'],
                    momentum=tunable_params['momentum'],
                    decay=tunable_params['decay'],
                    nesterov=tunable_params['nesterov']
                )
            elif optimizer_params.get('name') == 'RMSprop':
                tunable_params = {
                    'name': 'RMSprop',
                    'lr': optimizer_params.get('lr', 0.001),
                    'rho': optimizer_params.get('rho', 0.9),
                    'epsilon': optimizer_params.get('epsilon', None),
                    'decay': optimizer_params.get('decay', 0.0)
                }
                optimizer = RMSprop(
                    lr=tunable_params['lr'],
                    rho=tunable_params['rho'],
                    epsilon=tunable_params['epsilon'],
                    decay=tunable_params['decay']
                )
            elif optimizer_params.get('name') == 'Adam':
                tunable_params = {
                    'name': 'Adam',
                    'lr': optimizer_params.get('lr', 0.001)
                }
                optimizer = Adam(
                    lr=tunable_params['lr']
                )
            else:
                tunable_params = {
                    'name': 'Adadelta',
                    'lr': optimizer_params.get('lr', 0.1),
                    'rho': optimizer_params.get('rho', 0.95),
                    'epsilon': optimizer_params.get('epsilon', 1e-6),
                    'decay': optimizer_params.get('decay', 0.0)
                }
                # Default Adadelta
                optimizer = Adadelta(
                    lr=tunable_params['lr'],
                    rho=tunable_params['rho'],
                    epsilon=tunable_params['epsilon']
                )
            print('=== Optimization parameters ===')
            print(json.dumps(tunable_params, sort_keys=True, indent=4))
            print('=== Optimization parameters ===')
            return optimizer
        elif self.extractor_params['engine'] == 'pytorch':
            # TODO: make optimizer more configurable
            optimizer_params['name'] = optimizer_params.get('name', 'sgd')
            optimizer_params['lr'] = optimizer_params.get('lr', 0.3)
            optimizer_params['lr_decay'] = optimizer_params.get('lr_decay', 0.9)
            optimizer_params['decay_epoch'] = optimizer_params.get('decay_epoch', 5)
            return optimizer_params
        elif self.extractor_params['engine'] == 'transformers':
            pass
        else:
            raise Exception('Only Keras or PyTorch engines are supported.')

    def create_model(self):
        pass

    def __getstate__(self):
        u"""Defines what is to be pickled.
        Keras models cannot be pickled. Should call save_keras_model() and load_keras_model() separately.
        The sequence is :
        obj.save_keras_model('kerasFilename')
        pickle.dump(obj, fileHandle)
        ...
        obj = pickle.load(fileHandle)
        obj.load_keras_model()"""

        # Create state without self.keras_model
        state = dict(self.__dict__)
        #state.pop(u'keras_model')   # probably not needed anymore, now that we've made keras_model global
        return state

    def __setstate__(self, state):
        # Reload state for unpickling
        self.__dict__ = state

    def load_keras_model(self, filename=None):
        self.model = keras.models.load_model(filename, keras_custom_objects)

    def save_keras_model(self, filename):
        self.model.save(filename)
        print(self.model.summary())

    def predict(self, test_data_list):
        if 'engine' not in self.extractor_params or (('engine' in self.extractor_params) and (self.extractor_params['engine'] == 'keras')):
            return self.model.predict(test_data_list)
        elif self.extractor_params['engine'] == 'pytorch':
            from data.loader import DataLoader as BatchDataLoader
            print("Evaluating on test set...")
            predictions = []
            test_batch = BatchDataLoader(test_data_list, self.features.feature_strings, None, self.hyper_params.dict['batch_size'], self.hyper_params.dict, self.event_domain.event_roles, evaluation=True, test_mode=True)
            for i, batch in enumerate(test_batch):
                preds, _ = self.trained_model.predict(batch, compute_loss=False, compute_logits=True)
                predictions.append(preds)
            return np.vstack(predictions)
        else:
            raise Exception('Only Keras or PyTorch engines are supported.')