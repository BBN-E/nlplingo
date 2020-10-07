import codecs
import json
import os

import numpy as np
from nlplingo.nn.sequence_model import SequenceXLMRBase, SequenceXLMRCustom
from nlplingo.nn.spanpair_model import SpanPairModelEmbedded
from nlplingo.tasks.entitycoref.feature import EntityCorefFeatureGenerator
from nlplingo.tasks.entitycoref.generator import EntityCorefExampleGenerator
from nlplingo.tasks.eventcoref.feature import EventCorefFeatureGenerator
from nlplingo.tasks.eventcoref.generator import EventCorefExampleGenerator
from nlplingo.tasks.eventpair.feature import EventPairFeatureGenerator
from nlplingo.tasks.eventpair.generator import EventPairExampleGenerator
from nlplingo.tasks.eventframe.feature import EventFramePairFeatureGenerator
from nlplingo.tasks.eventframe.generator import EventFramePairExampleGenerator

from keras.models import load_model as keras_load_model
from keras.models import Model as KerasModel

from nlplingo.tasks.eventargument.feature import EventArgumentFeatureGenerator
from nlplingo.tasks.eventargument.generator import EventArgumentExampleGenerator
from nlplingo.tasks.eventrelation.feature import EventEventRelationFeatureGenerator
from nlplingo.tasks.eventrelation.generator import EventEventRelationExampleGenerator
from nlplingo.tasks.entityrelation.feature import EntityRelationFeatureGenerator
from nlplingo.tasks.entityrelation.generator import EntityRelationExampleGenerator

from nlplingo.tasks.event_domain import EventDomain
from nlplingo.tasks.eventtrigger.feature import EventTriggerFeatureGenerator
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator

from nlplingo.nn.argument_model import CNNArgumentModel
from nlplingo.nn.argument_model import GNNArgumentModel
from nlplingo.nn.argument_model import MultiLayerArgumentModelEmbedded, WithinSentenceArgumentModel
from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.common import keras_custom_objects
from nlplingo.nn.trigger_model import CNNTriggerModel
from nlplingo.nn.trigger_model import MultiLayerTriggerModelEmbedded
from nlplingo.nn.eventpair_model import EventPairModelEmbeddedTrigger
from nlplingo.nn.event_event_relation_model import MultiLayerEventEventRelationModel, WithinSentenceEER, EventEventRelationStackedOpenNREModel, WithinSentenceEERGeneral
from nlplingo.nn.entity_entity_relation_model import MultiLayerEntityRelationModelEmbedded, WithinSentenceEntityRelationModel
from nlplingo.nn.eventframepair_model import EventFramePairModelEmbedded

from nlplingo.nn.hyperparameters import HyperParameters
from nlplingo.tasks.common.feature.feature_setting import FeatureSetting

import logging

from nlplingo.tasks.sequence.feature import SequenceFeatureGenerator
from nlplingo.tasks.sequence.generator import SequenceExampleGenerator

logger = logging.getLogger(__name__)


class Extractor(object):
    trigger_model_table = {
        'event-trigger_cnn': CNNTriggerModel,
        'event-trigger_cnn-embedded': CNNTriggerModel,
        'event-trigger_multilayer-embedded': MultiLayerTriggerModelEmbedded,
        'event-trigger_sentence-embedded': MultiLayerTriggerModelEmbedded,
    }

    argument_model_table = {
        'event-argument_cnn': CNNArgumentModel,
        'event-argument_cnn-embedded': CNNArgumentModel,
        'event-argument_gnn': GNNArgumentModel,
        'event-argument_multilayer-embedded': MultiLayerArgumentModelEmbedded,
        'event-argument_bert-mention': WithinSentenceArgumentModel
    }

    eer_model_table = {
        'event-event-relation_multilayer': MultiLayerEventEventRelationModel,
        'event-event-relation_multilayer-embedded': MultiLayerEventEventRelationModel,
        'event-event-relation_two_models_with_postprocessing': EventEventRelationStackedOpenNREModel,
        'event-event-relation_cnn-embedded': WithinSentenceEER, # This exists for legacy reasons
        'event-event-relation_within-sentence': WithinSentenceEER,
        'event-event-relation_general_decode-embedded': WithinSentenceEERGeneral
    }

    entity_relation_model_table = {
        'entity-entity-relation_multilayer-embedded': MultiLayerEntityRelationModelEmbedded,
        'entity-entity-relation_bert-mention': WithinSentenceEntityRelationModel
    }

    eventpair_model_table = {
        'event-pair_embedded': SpanPairModelEmbedded,
        'event-pair_embedded_trigger': EventPairModelEmbeddedTrigger
    }

    eventframepair_model_table = {
        'event-framepair_embedded': EventFramePairModelEmbedded
    }

    entitycoref_model_table = {
        'entitycoref_embedded': SpanPairModelEmbedded
    }

    eventcoref_model_table = {
        'eventcoref_embedded': SpanPairModelEmbedded
    }

    sequence_model_table = {
        'sequence_xlmr-base': SequenceXLMRBase,
        'sequence_xlmr-custom': SequenceXLMRCustom
    }


    def __init__(self, params, extractor_params, embeddings, load_from_file=False):
        """
        :type params: dict              # general parameters
        :type extractor_params: dict    # specific to this extractor
        :type embeddings: dict[str : nlplingo.embeddings.word_embeddings.WordEmbedding]
        """

        self.extractor_params = extractor_params
        self.extractor_name = extractor_params.get('extractor_name', None)

        self.task = extractor_params.get('task', None)
        self.engine = extractor_params.get('engine', None)

        self.model_type = extractor_params['model_type']
        """:type: str"""

        self.domain = EventDomain.read_domain_ontology_file(extractor_params['domain_ontology'],
                                                            domain_name=extractor_params.get('domain_name', 'general'))
        """:type: nlplingo.tasks.event_domain.EventDomain"""
        self.domain.build_prior(extractor_params.get('ontology_yaml'))

        self.model_file = extractor_params['model_file']
        """:type: str"""

        self.class_thresholds_path = extractor_params.get('class_thresholds')
        """:type: str"""

        self.class_thresholds_global = float(
            extractor_params.get('class_thresholds_global', -1.0))
        """:type: float"""

        self.use_trigger_safelist = extractor_params.get('trigger.use_safelist', False)

        if 'engine' not in extractor_params or (('engine' in extractor_params) and (extractor_params['engine'] == 'keras')):
            self.hyper_parameters = HyperParameters(extractor_params['hyper-parameters'], load_from_file)
        elif extractor_params['engine'] == 'pytorch':
            self.hyper_parameters = HyperParameters(extractor_params['hyper-parameters'], load_from_file)
        # elif extractor_params['engine'] == 'transformers':
        #     pass
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(self.model_type))

        """:type: nlplingo.nn.extractor.HyperParameters"""

        self.feature_setting = FeatureSetting(self.extractor_params['features'])

        self.extraction_model = None
        if self.model_type in self.trigger_model_table:
            self.extraction_model = self.trigger_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting)
        elif self.model_type in self.argument_model_table:
            self.extraction_model = self.argument_model_table[self.model_type](params, extractor_params, self.domain,
                                                                               embeddings, self.hyper_parameters,
                                                                               self.feature_setting)
        elif self.model_type in self.eventpair_model_table:
            self.extraction_model = self.eventpair_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting) # TODO: fix this model
        elif self.model_type in self.eer_model_table:
            self.extraction_model = self.eer_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting)
        elif self.model_type in self.entity_relation_model_table:
            self.extraction_model = self.entity_relation_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                                      self.hyper_parameters,
                                                                                      self.feature_setting)
        elif self.model_type in self.eventframepair_model_table:
            self.extraction_model = self.eventframepair_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting) # TODO: fix this model
        elif self.model_type in self.entitycoref_model_table:
            self.extraction_model = self.entitycoref_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting)
        elif self.model_type in self.eventcoref_model_table:
            self.extraction_model = self.eventcoref_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_setting)
        elif self.model_type in self.sequence_model_table:
            if self.task == 'event-trigger':
                self.domain.create_sequence_types(self.domain.event_types)
            elif self.task == 'event-argument':
                self.domain.create_sequence_types(self.domain.event_roles)
            elif self.task == 'ner':
                self.domain.create_sequence_types(self.domain.entity_types)

            self.extraction_model = self.sequence_model_table[self.model_type](params, extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters, self.feature_setting)
        elif self.model_type.startswith('oregon'):  # TODO hack, until YS has time to properly integrate after BETTER eval
            pass
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(self.model_type))

        """:type: nlplingo.nn.event_model.ExtractionModel"""
        # TODO: extend this to support EventEventRelation models
        if load_from_file:
            logging.info('Loading previously trained model')

            if extractor_params.get('engine', None) == 'keras':
                self.load_keras()
            if extractor_params.get('engine', None) is None:  # TODO use framework
                self.load_keras()
            elif extractor_params['engine'] == 'pytorch':
                pass
            # elif extractor_params['engine'] == 'transformers':
            #     pass
            else:
                raise Exception(
                    'Only Keras or PyTorch engines are supported.')

        #if ('engine' in extractor_params) and (extractor_params['engine'] == 'pytorch'):
        #    if load_from_file or self.extraction_model.hyper_params.load:
        #        pass
                """
                self.extraction_model.hyper_params.num_class = self.extraction_model.num_output
                if self.extraction_model.word_embeddings is not None:
                    trainer = self.extraction_model.model(self.extraction_model.extractor_params, self.extraction_model.hyper_params.dict, self.extraction_model.optimizer,
                                         feature_names=self.extraction_model.features.feature_strings, emb_matrix=self.extraction_model.word_embeddings)
                else:  # frozen, external embedding case
                    if self.extraction_model.embeddings_vector_size is not None:
                        self.extraction_model.hyper_params.dict['emb_dim'] = self.extraction_model.embeddings_vector_size
                        trainer = self.extraction_model.model(self.extraction_model.extractor_params, self.extraction_model.hyper_params.dict, self.extraction_model.optimizer,
                                             feature_names=self.extraction_model.features.feature_strings)

                if self.model_file:
                    trainer.load(self.model_file)
                    self.extraction_model.trained_model = trainer
                """

        self.feature_generator = None  # feature generator
        self.example_generator = None  # example generator

        # TODO this should really be renamed as task instead of model_type
        if self.model_type.startswith('event-trigger_'):
            self.feature_generator = EventTriggerFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting, self.domain)
            self.example_generator = EventTriggerExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('event-argument_'):
            self.feature_generator = EventArgumentFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting)
            self.example_generator = EventArgumentExampleGenerator(self.domain, params, extractor_params,
                                                                   self.hyper_parameters)
        elif self.model_type.startswith('event-pair_'):
            self.feature_generator = EventPairFeatureGenerator(extractor_params)
            self.example_generator = EventPairExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('event-event-relation_'):
            self.feature_generator = EventEventRelationFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting)
            self.example_generator = EventEventRelationExampleGenerator(self.domain, params, extractor_params,
                                                               self.hyper_parameters)
        elif self.model_type.startswith('entity-entity-relation_'):
            self.feature_generator = EntityRelationFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting)
            self.example_generator = EntityRelationExampleGenerator(self.domain, params, extractor_params,
                                                               self.hyper_parameters)
        elif self.model_type.startswith('event-framepair_'):
            self.feature_generator = EventFramePairFeatureGenerator(extractor_params)
            self.example_generator = EventFramePairExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('entitycoref_'):
            self.feature_generator = EntityCorefFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting)
            self.example_generator = EntityCorefExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('eventcoref_'):
            self.feature_generator = EventCorefFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting)
            self.example_generator = EventCorefExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('oregon'):      # TODO hack, until YS has time to properly integrate after BETTER eval
            pass
        elif self.model_type.startswith('sequence_'):
            self.feature_generator = SequenceFeatureGenerator(extractor_params, self.hyper_parameters, self.feature_setting, self.extraction_model.tokenizer, self.domain)
            self.example_generator = SequenceExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(self.model_type))

        self.extraction_model_last_layer = None
        """:type: keras.models.Model"""
        self.emit_vectors = extractor_params.get('output_vectors', False)

        self.class_thresholds = None
        # load saved thresholds from file
        self._build_threshold_vector()

        # use a global threshold value if they were not loaded
        if self.class_thresholds is None:
            logging.info('Using global threshold override for {}'.format(
                    self.extractor_name))

            # use defaults, if no global override given in extractor parameters
            if self.class_thresholds_global < 0.0:
                logging.info('Using default thresholds for {}'.format(
                    self.extractor_name))
                self.class_thresholds_global = 0.5
            number_of_classes = len(self.domain.event_types.keys())

            logging.info('- global threshold ={}'.format(self.class_thresholds_global))
            self.class_thresholds = np.asarray(
                [self.class_thresholds_global] * number_of_classes)

    def _build_threshold_vector(self):
        path = self.class_thresholds_path
        if path is not None and os.path.isfile(str(path)):
            if path.endswith('.npz'):
                self.class_thresholds = np.load(str(path))['thresholds']
                print('Loaded saved thresholds from NPZ for {}'.format(
                    self.extractor_name))
            elif path.endswith('.json'):
                number_of_classes = len(self.domain.event_types.keys())
                self.class_thresholds = np.asarray([0.5] * number_of_classes)
                with codecs.open(path, 'r', encoding='utf8') as fh:
                    thresholds_json = json.load(fh)
                for label, threshold in thresholds_json.items():
                    try:
                        index = self.domain.get_event_type_index(label)
                        self.class_thresholds[index] = float(threshold)
                    except ValueError as e:
                        print('The following error occurred while loading '
                              'thresholds from json and will be ignored:\n'
                              '{}'.format(e))
                print('Loaded saved thresholds from JSON for {}'.format(
                    self.extractor_name))

    def make_last_layer_model(self):

        if self.extraction_model_last_layer is not None:
            print("Last layer of model has already been built")
            return

        keras_model = self.extraction_model
        if type(keras_model) is not KerasModel:
            keras_model = keras_model.model
            """:type: keras.models.Model"""
        print("Original model:")
        try:
            print(keras_model.summary())
        except TypeError:
            print("Keras encountered an error when trying to print the model "
                  "summary.  Skipping this printout...")

        self.extraction_model_last_layer = KerasModel(
            inputs=keras_model.input,
            outputs=keras_model.layers[-2].output)

        print("Copy model:")
        try:
            print(self.extraction_model_last_layer.summary())
        except TypeError:
            print("Keras encountered an error when trying to print the copy's "
                  "summary.  Skipping this printout...")

    def get_embeddings(self, examples, data_list):
        ret = []
        self.make_last_layer_model()
        vectors = self.extraction_model_last_layer.predict(data_list)
        for i, ex in enumerate(examples):
            output_vector = vectors[i, :]
            ret.append(output_vector)
        return ret

    def load_keras(self):
        try:
            trained_keras_model = keras_load_model(self.model_file)
        except ValueError:
            custom_objects = keras_custom_objects
            trained_keras_model = keras_load_model(self.model_file, custom_objects)

        weights = trained_keras_model.get_weights()
        new_weights = []
        for i, w in enumerate(weights):
            pretrained = self.extraction_model.layers.pretrained_embeddings
            using_pretrained = pretrained is not None
            if using_pretrained and i > 1 and w.shape[0] == pretrained.shape[0]:
                # TODO retrain models to avoid this hack
                pass
            else:
                new_weights.append(w)
        weights = new_weights
        # for i, w in enumerate(weights):
        #     print(i, w.shape


        self.extraction_model.model.set_weights(weights)

