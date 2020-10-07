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

from nlplingo.tasks.eventrelation.postprocess import postfilter, construct_eer_key, postfilter_general
from nlplingo.tasks.eventrelation.example import EventEventRelationExample

from nlplingo.nn.pytorch_models.encoder.cnn_encoder import CNNEncoder, CNNEncoderLegacy, CNNEncoderLegacyDecode
from nlplingo.nn.pytorch_models.model.softmax_nn import SoftmaxNN, SoftmaxNNLegacy, SoftmaxNNLegacyMentionPool
from nlplingo.nn.pytorch_models.encoder.bert_encoder import BERTEncoder, BERTEncoderTrain, BERTEncoderMentionPool, BERTEncoderMentionPoolDecode
from nlplingo.nn.framework.sentence_re import SentenceREDecode
import torch
import json
import time
import nlplingo.nn.constants

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


class EventEventRelationModel(ExtractionModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(EventEventRelationModel, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)

        self.num_output = len(eer_domain.eer_types)

    @property
    def none_label_index(self):
        return self.event_domain.get_eer_type_index('NA')


class EventEventRelationKerasModel(EventEventRelationModel, KerasExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type params: dict
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        # Calls EventEventRelationModel init (for task-specific model params)
        # then KerasExtractionModel init (builds Keras LayerCreator using them)
        super(EventEventRelationKerasModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)

    def create_model(self):
        super(EventEventRelationKerasModel, self).create_model()


LEARNIT_GIGAWORD_THRESHOLD = .7
LDC_THRESHOLD = .7


class EventEventRelationStackedOpenNREModel(EventEventRelationModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(EventEventRelationStackedOpenNREModel, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)
        # TODO: fix for case where each extractor has different embeddings
        # Create sub extractors
        from nlplingo.nn.extractor import Extractor
        self.sub_extractors = []
        sub_extractors_params = extractor_params.get('sub_extractors', None)
        start = time.time()
        for extractor_params in sub_extractors_params:
            extractor = Extractor(params, extractor_params, embeddings, True)
            self.sub_extractors.append(extractor)
        end = time.time()
        logging.info('Models loaded in %s seconds', end - start)

    def predict_triplets(self, eer_examples, decode_triplets, all_eer_predictions):
        # arbitrary convention that LDC model appears first
        #curr_indices_ldc = list(range(len(eer_examples)))
        #curr_indices_giga = list(range(len(eer_examples)))

        all_eer_predictions = self.sub_extractors[0].extraction_model.predict_triplets(eer_examples, decode_triplets, all_eer_predictions)
        all_eer_predictions = self.sub_extractors[1].extraction_model.predict_triplets(eer_examples, decode_triplets, all_eer_predictions)

        #for internal_idx, orig_idx in enumerate(eer_order_transform_0):
        #    assert(e0[internal_idx] is eer_examples[orig_idx])

        #for internal_idx, orig_idx in enumerate(eer_order_transform_1):
        #    assert(e1[internal_idx] is eer_examples[orig_idx])
        # arbitrary convention that LearnIt Gigaword model appears second
        logging.info('all_eer_predictions: %s', len(all_eer_predictions))
        return all_eer_predictions
        # exit(0)
        # start = time.time()
        # common_set = set(eer_order_transform_0).intersection(set(eer_order_transform_1))

        #idx = 0
        #for idx, (example, prediction, conf) in enumerate(zip(e0,p0,c0)):
        #    eer_key = example.sentence.docid + '#' + str(example.sentence.index) + '#' + example.anchor1.int_pair.to_string() + '#' + examples.anchor2.int_pair.to_string()
        """
        if len(common_set) > 0 :
            eer_transform_0_inv = dict()
            for i_idx, a_idx in enumerate(eer_order_transform_0):
                eer_transform_0_inv[a_idx] = i_idx
            eer_transform_1_inv = dict()
            for i_idx, a_idx in enumerate(eer_order_transform_1):
                eer_transform_1_inv[a_idx] = i_idx
            print('Common example number', len(common_set))
            for idx in common_set:
                print('Example', decode_triplets[idx])
                print('Prediction LDC', p0[eer_transform_0_inv[idx]], c0[eer_transform_0_inv[idx]])
                print('Prediction Learnit', p1[eer_transform_1_inv[idx]], c1[eer_transform_1_inv[idx]])
            # exit(0)
        """
        #total_relations = 0

        #eer_examples_pt_all = list()
        #predicted_role_all = list()
        #confidences = list()

        """
        head_tail_dict = dict()
        # fusion
        PREF_LDC_OVER_LEARNIT_GIGAWORD = True
        if PREF_LDC_OVER_LEARNIT_GIGAWORD:
            for idx in range(len(p0)):
                example = e0[idx]
                eer_key = construct_eer_key(example)
                if eer_key not in head_tail_dict:
                    head_tail_dict[eer_key] = []
                head_tail_dict[eer_key].append((c0[idx], p0[idx], example, 'ldc'))
                # eer_examples_pt_all.append(example)
                # predicted_role_all.append(p0[idx])
                # confidences.append(c0[idx])
            for idx in range(len(p1)):
                if eer_order_transform_1[idx] not in common_set:
                    # eg = e1[idx]
                    # assert isinstance(eg, EventEventRelationExample)
                    example = e1[idx]
                    eer_key = construct_eer_key(example)
                    if eer_key not in head_tail_dict:
                        head_tail_dict[eer_key] = []
                    head_tail_dict[eer_key].append((c1[idx], p1[idx], example, 'giga'))
                    # eer_examples_pt_all.append(e1[idx])
                    # predicted_role_all.append(p1[idx])
                    # confidences.append(c1[idx])
        else:
            eer_examples_pt_all.extend(e1)
            predicted_role_all.extend(p1)
            confidences.extend(c1)
            for idx in range(len(p0)):
                if eer_order_transform_0[idx] not in common_set:
                    # eg = e0[idx]
                    # assert isinstance(eg, EventEventRelationExample)
                    eer_examples_pt_all.append(e0[idx])
                    predicted_role_all.append(p0[idx])
                    confidences.append(c0[idx])
        end = time.time()
        logging.info('Fusion for two models and LearnIt relations took %s seconds', end - start)
        logging.debug('Total_relations from LDC model: %s', len(p0))
        logging.debug('Total_relations from Gigaword model: %s', len(predicted_role_all) - len(p0))
        """
        """
        print('final prediction size', len(predicted_role_all))
        print('before pruning size', len(p0) + len(p1))
        """
        # return (eer_examples_pt_all, predicted_role_all, confidences), head_tail_dict
        # return e0 + e1, p0 + p1, c0 + c1 #, # eer_examples_pt_all, predicted_role_all, confidences

class WithinSentenceEER(EventEventRelationModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(WithinSentenceEER, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)
        self.create_model()
        self.none_class_index = self.event_domain.get_eer_type_index('NA')

    def create_model(self):
        logging.info('Initializing the sentence encoder.')
        rel2id = self.event_domain.eer_types
        word2id = None
        word2vec = None
        # Load a model for decoding
        nlplingo.nn.constants.change_global_max_sent_length(self.hyper_params.max_sentence_length)
        if self.hyper_params.decode_mode:
            if not hasattr(self.hyper_params, 'absolute_ckpt'): # hacky
                #rel2id = json.load(open(os.path.join(self.hyper_params.opennre_rootpath, 'benchmark/{}/{}_rel2id.json'
                #                                     .format(self.hyper_params.opennre_dataset, self.hyper_params.opennre_dataset))))
                # TODO: get rid of this rel2id assignment. make sure to create an ontology for learnit-gigaword
                opennre_ckpt = os.path.join(self.hyper_params.opennre_rootpath + '/benchmark/' + self.hyper_params.opennre_dataset,
                                            'ckpt/{}'.format(self.hyper_params.opennre_ckpt))
            else:
                opennre_ckpt = self.hyper_params.opennre_ckpt

            if self.hyper_params.encoder == 'cnn':
                # TODO: phase the rel2id out, use nlplingo-native embeddings loader
                word2id = json.load(open(os.path.join(self.hyper_params.opennre_rootpath,
                                                      'pretrain/glove/glove.6B.50d_word2id.json')))
                word2vec = np.load(os.path.join(self.hyper_params.opennre_rootpath,
                                                'pretrain/glove/glove.6B.50d_mat.npy'))
                sentence_encoder = CNNEncoderLegacy(
                    token2id=word2id, max_length=self.hyper_params.max_sentence_length, word_size=self.embeddings_vector_size, position_size=self.hyper_params.position_embedding_vector_length,
                    hidden_size=230, blank_padding=True, kernel_size=3, padding_size=1,
                    word2vec=word2vec, dropout=self.hyper_params.dropout)
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'cnn_opt':
                # TODO: phase the rel2id out, use nlplingo-native embeddings loader
                word2id = json.load(open(os.path.join(self.hyper_params.opennre_rootpath,
                                                      'pretrain/glove/glove.6B.50d_word2id.json')))
                word2vec = np.load(os.path.join(self.hyper_params.opennre_rootpath,
                                                'pretrain/glove/glove.6B.50d_mat.npy'))
                nlplingo.nn.constants.change_global_blank_id(word2id['[PAD]'])
                sentence_encoder = CNNEncoderLegacyDecode(
                    token2id=word2id, max_length=self.hyper_params.max_sentence_length,
                    word_size=self.embeddings_vector_size,
                    position_size=self.hyper_params.position_embedding_vector_length,
                    hidden_size=230, blank_padding=True, kernel_size=3, padding_size=1,
                    word2vec=word2vec, dropout=self.hyper_params.dropout)
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'bert_original' or self.hyper_params.encoder == 'bert_legacy':
                sentence_encoder = BERTEncoder(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False, encoder=self.hyper_params.encoder
                )
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'distilbert_original':
                sentence_encoder = BERTEncoder(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False, encoder=self.hyper_params.encoder
                )
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'bert_mention':
                sentence_encoder = BERTEncoderMentionPoolDecode(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False, encoder=self.hyper_params.encoder, is_decode=True
                )
                self.model = SoftmaxNNLegacyMentionPool(sentence_encoder, len(rel2id), rel2id, hidden_factor_size=2)

            if self.extractor_params['cpu']:
                self.model.load_state_dict(torch.load(opennre_ckpt, map_location=torch.device('cpu'))['state_dict'])
            else:
                print('Loading model at', (opennre_ckpt))
                self.model.load_state_dict(torch.load(opennre_ckpt)['state_dict'])
        else: # train mode
            if self.hyper_params.encoder == 'cnn':
                # TODO: phase the rel2id out, use nlplingo-native embeddings loader
                word2id = json.load(open(os.path.join(self.hyper_params.opennre_rootpath,
                                                      'pretrain/glove/glove.6B.50d_word2id.json')))
                word2vec = np.load(os.path.join(self.hyper_params.opennre_rootpath,
                                                'pretrain/glove/glove.6B.50d_mat.npy'))
                sentence_encoder = CNNEncoderLegacy(
                    token2id=word2id, max_length=self.hyper_params.max_sentence_length, word_size=self.embeddings_vector_size, position_size=self.hyper_params.position_embedding_vector_length,
                    hidden_size=230, blank_padding=True, kernel_size=3, padding_size=1,
                    word2vec=word2vec, dropout=self.hyper_params.dropout)
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'bert' or self.hyper_params.encoder == 'distilbert':
                sentence_encoder = BERTEncoderTrain(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False
                )
                self.model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'bert_original' or self.hyper_params.encoder == 'distilbert_original':
                sentence_encoder = BERTEncoder(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False, encoder=self.hyper_params.encoder, is_decode=False
                )
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id)
            elif self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'albert_mention':
                sentence_encoder = BERTEncoderMentionPool(
                    max_length=self.hyper_params.max_sentence_length,
                    pretrain_path=os.path.join(self.hyper_params.opennre_rootpath, 'pretrain/bert-base-uncased'),
                    mask_entity=False, encoder=self.hyper_params.encoder, is_decode=False
                )
                self.model = SoftmaxNNLegacy(sentence_encoder, len(rel2id), rel2id, hidden_factor_size=2)

    # TODO: this function can be further optimized in the mention_pool case by
    # storing decode_triplets as pure offset information, relegating the sentence text to another data structure
    def predict_triplets(self, eer_examples, decode_triplets, all_eer_predictions):
        # num_unique_sent = len(sent_eer_dict)
        # print('Number of unique sentences', num_unique_sent)
        # print('eer_examples original', len(eer_examples))
        eer_order_transform = list(range(len(eer_examples)))

        # Organize the sentence/head/tail offset triplets into
        # lists organized by sentence text and their associated offsets
        if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
            sent_id_to_sent_map = {}
            sent_id_to_head_map = {}
            sent_id_to_tail_map = {}
            sent_id_to_idx_map = {}

            for idx, eer_example in enumerate(eer_examples):
                sent_id = eer_example.sentence.docid + '#' + str(eer_example.sentence.index)
                if sent_id not in sent_id_to_sent_map:
                    sent_id_to_sent_map[sent_id] = decode_triplets[idx]['text']
                    sent_id_to_head_map[sent_id] = []
                    sent_id_to_tail_map[sent_id] = []
                    sent_id_to_idx_map[sent_id] = []
                sent_id_to_head_map[sent_id].append(decode_triplets[idx]['h']['pos'])
                sent_id_to_tail_map[sent_id].append(decode_triplets[idx]['t']['pos'])
                sent_id_to_idx_map[sent_id].append(idx)

            items = []
            global_eer_order = []
            for sent_id in sent_id_to_sent_map:
                item = [sent_id_to_sent_map[sent_id], sent_id_to_head_map[sent_id], sent_id_to_tail_map[sent_id]]
                items.append(item)
                global_eer_order.extend(sent_id_to_idx_map[sent_id])
            decode_triplets = items
            eer_examples = [eer_examples[i] for i in global_eer_order]
            eer_order_transform = [eer_order_transform[i] for i in global_eer_order]
        #print('here', len(eer_examples))

        if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
            optimize_batches = True
        else:
            optimize_batches = False

        if optimize_batches:
            infer_framework1 = SentenceREDecode(self.model, decode_triplets,
                                                self.extractor_params,
                                                self.hyper_params,
                                                self.event_domain.eer_types, optimize_batches=True)
        else:
            infer_framework1 = SentenceREDecode(self.model, decode_triplets,
                                                self.extractor_params,
                                                self.hyper_params,
                                                self.event_domain.eer_types, optimize_batches=False)

        start = time.time()
        logging.info('Start of EER predicting for single model')
        if optimize_batches:
            ldc_eer_predictions = infer_framework1.eval_model()
            if hasattr(infer_framework1, 'delete_indices'):
                tmp_list = []
                eer_order_tmp = []
                for idx, example in enumerate(eer_examples):
                    #print('delete_indices', infer_framework1.delete_indices)
                    #print('delete_indices', len(infer_framework1.delete_indices))
                    if idx not in infer_framework1.delete_indices:
                        tmp_list.append(example)
                        eer_order_tmp.append(eer_order_transform[idx])
                eer_examples = tmp_list
                eer_order_transform = eer_order_tmp
                # print('here2', len(eer_examples))
            eer_examples = [eer_examples[i] for i in infer_framework1.sort_order]
            eer_order_transform = [eer_order_transform[i] for i in infer_framework1.sort_order]
            #print('here3', len(eer_examples))
        else:
            ldc_eer_predictions = infer_framework1.eval_model()
        # print(len(eer_examples))
        # print(len(ldc_eer_predictions))
        assert(len(ldc_eer_predictions) == len(eer_examples))
        end = time.time()
        logging.info('EER predictions only for single model took %s seconds', end - start)

        start = time.time()
        logging.info('Start of postfiltering for single model')
        curr_indices_ldc = list(range(len(eer_examples)))

        if self.hyper_params.encoder == 'cnn' or self.hyper_params.encoder == 'cnn_opt':
            threshold = LEARNIT_GIGAWORD_THRESHOLD
        elif self.hyper_params.encoder == 'bert_mention':
            threshold = LDC_THRESHOLD
        else:
            raise Exception('encoder not supported for postfiltering')
        postfilter(curr_indices_ldc, eer_examples, ldc_eer_predictions, self.event_domain, self.none_class_index, threshold, all_eer_predictions, self.hyper_params.encoder)
        end = time.time()
        logging.info('Postfiltering only for single model took %s seconds', end - start)

        total_relations = 0

        eer_examples_pt_all = list()
        predicted_role_all = list()
        confidences = list()

        """
        for idx, prediction in enumerate(ldc_eer_predictions):
            if idx in curr_indices_ldc:
                index = self.event_domain.get_eer_type_index(prediction[0])
                total_relations += 1
                eg = eer_examples[idx]
                assert isinstance(eg, EventEventRelationExample)
                predicted_role = self.event_domain.get_eer_type_from_index(index)
                eer_examples_pt_all.append(eg)
                predicted_role_all.append(predicted_role)
                confidences.append(prediction[1])
        # print('total_relations', total_relations)
        """
        eer_order_transform = [eer_order_transform[i] for i in curr_indices_ldc]
        return all_eer_predictions
        # return (eer_examples_pt_all, predicted_role_all, confidences, None), eer_order_transform


class WithinSentenceEERGeneral(WithinSentenceEER):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(WithinSentenceEER, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)
        self.create_model()
        self.none_class_index = self.event_domain.get_eer_type_index('NA')

    # TODO: this function can be further optimized in the mention_pool case by
    # storing decode_triplets as pure offset information, relegating the sentence text to another data structure
    def predict_triplets(self, eer_examples, decode_triplets, all_eer_predictions):
        # num_unique_sent = len(sent_eer_dict)
        # print('Number of unique sentences', num_unique_sent)
        # print('eer_examples original', len(eer_examples))
        eer_order_transform = list(range(len(eer_examples)))

        # Organize the sentence/head/tail offset triplets into
        # lists organized by sentence text and their associated offsets
        if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
            sent_id_to_sent_map = {}
            sent_id_to_head_map = {}
            sent_id_to_tail_map = {}
            sent_id_to_idx_map = {}

            for idx, eer_example in enumerate(eer_examples):
                sent_id = eer_example.sentence.docid + '#' + str(eer_example.sentence.index)
                if sent_id not in sent_id_to_sent_map:
                    sent_id_to_sent_map[sent_id] = decode_triplets[idx]['text']
                    sent_id_to_head_map[sent_id] = []
                    sent_id_to_tail_map[sent_id] = []
                    sent_id_to_idx_map[sent_id] = []
                sent_id_to_head_map[sent_id].append(decode_triplets[idx]['h']['pos'])
                sent_id_to_tail_map[sent_id].append(decode_triplets[idx]['t']['pos'])
                sent_id_to_idx_map[sent_id].append(idx)

            items = []
            global_eer_order = []
            for sent_id in sent_id_to_sent_map:
                item = [sent_id_to_sent_map[sent_id], sent_id_to_head_map[sent_id], sent_id_to_tail_map[sent_id]]
                items.append(item)
                global_eer_order.extend(sent_id_to_idx_map[sent_id])
            decode_triplets = items
            eer_examples = [eer_examples[i] for i in global_eer_order]
            eer_order_transform = [eer_order_transform[i] for i in global_eer_order]

        if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
            optimize_batches = True
        else:
            optimize_batches = False

        if optimize_batches:
            infer_framework1 = SentenceREDecode(self.model, decode_triplets,
                                                self.extractor_params,
                                                self.hyper_params,
                                                self.event_domain.eer_types, optimize_batches=True)
        else:
            infer_framework1 = SentenceREDecode(self.model, decode_triplets,
                                                self.extractor_params,
                                                self.hyper_params,
                                                self.event_domain.eer_types, optimize_batches=False)

        start = time.time()
        logging.info('Start of EER predicting for single model')
        if optimize_batches:
            ldc_eer_predictions = infer_framework1.eval_model()
            if hasattr(infer_framework1, 'delete_indices'):
                tmp_list = []
                eer_order_tmp = []
                for idx, example in enumerate(eer_examples):
                    #print('delete_indices', infer_framework1.delete_indices)
                    #print('delete_indices', len(infer_framework1.delete_indices))
                    if idx not in infer_framework1.delete_indices:
                        tmp_list.append(example)
                        eer_order_tmp.append(eer_order_transform[idx])
                eer_examples = tmp_list
                eer_order_transform = eer_order_tmp
                # print('here2', len(eer_examples))
            eer_examples = [eer_examples[i] for i in infer_framework1.sort_order]
            eer_order_transform = [eer_order_transform[i] for i in infer_framework1.sort_order]
            #print('here3', len(eer_examples))
        else:
            ldc_eer_predictions = infer_framework1.eval_model()
        # print(len(eer_examples))
        # print(len(ldc_eer_predictions))
        assert(len(ldc_eer_predictions) == len(eer_examples))
        end = time.time()
        logging.info('EER predictions only for single model took %s seconds', end - start)

        start = time.time()
        logging.info('Start of postfiltering for single model')
        curr_indices_ldc = list(range(len(eer_examples)))

        postfilter_general(curr_indices_ldc, eer_examples, ldc_eer_predictions, self.event_domain, self.none_class_index, self.hyper_params.decoding_threshold, all_eer_predictions, self.hyper_params.encoder)
        end = time.time()
        logging.info('Postfiltering only for single model took %s seconds', end - start)

        eer_order_transform = [eer_order_transform[i] for i in curr_indices_ldc]
        return all_eer_predictions


class MultiLayerEventEventRelationModel(EventEventRelationKerasModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(MultiLayerEventEventRelationModel, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        super(MultiLayerEventEventRelationModel, self).create_model()
        model_input_dict = dict()
        outputs_to_merge = []

        # one-hot word vectors in binary datapoint window
        self.layers.add_binary_window_layer(
            "arg0_arg1_window", model_input_dict, outputs_to_merge,
            self.layers.EmbeddingLayer.PRETRAINED)

        # dense word vectors in binary datapoint window
        self.layers.add_binary_window_layer(
            "arg0_arg1_window_vector", model_input_dict, outputs_to_merge,
            self.layers.EmbeddingLayer.NONE)

        # hidden dense layers
        hidden_input = self.layers.merge(outputs_to_merge)
        hidden_output = self.layers.build_hidden_layers(hidden_input)

        # add event embeddings after hidden layers
        to_output_layer_list = [hidden_output]
        self.layers.add_event_embedding_layer(
            "event_embeddings", model_input_dict, to_output_layer_list, with_dropout=True)

        # historically self.activation = 'softmax'
        model_outputs = []
        self.layers.add_decision_layer(to_output_layer_list, model_outputs, dropout=False)

        self.compile(model_outputs, model_input_dict)
