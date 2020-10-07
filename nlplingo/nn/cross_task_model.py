from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
import logging

from keras.layers import Flatten, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.constraints import maxnorm
from keras.layers import concatenate

from keras.callbacks import EarlyStopping

from nlplingo.nn.extraction_model import ExtractionModel

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

LEARNIT_GIGAWORD_THRESHOLD = .7
LDC_THRESHOLD = .7

class WithinSentence(ExtractionModel):
    def __init__(self, params, extractor_params, eer_domain, embeddings, hyper_params, features):
        super(WithinSentence, self).__init__(params, extractor_params, eer_domain, embeddings, hyper_params, features)

        # self.event_domain_rel = None
        # self.create_model()
        # self.none_class_index = self.event_domain.get_eer_type_index('NA')

    def create_model(self):
        logging.info('Initializing the sentence encoder.')
        rel2id = self.event_domain_rel
        word2id = None
        word2vec = None
        # Load a model for decoding
        nlplingo.nn.constants.change_global_max_sent_length(self.hyper_params.max_sentence_length)
        if self.hyper_params.decode_mode:
            if not hasattr(self.hyper_params, 'absolute_ckpt'): # hacky
                opennre_ckpt = os.path.join(self.hyper_params.opennre_rootpath + '/benchmark/' + self.hyper_params.opennre_dataset,
                                            'ckpt/{}'.format(self.hyper_params.opennre_ckpt))
            else:
                opennre_ckpt = self.hyper_params.opennre_ckpt

            if self.hyper_params.encoder == 'cnn':
                # use nlplingo-native embeddings loader
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
                # use nlplingo-native embeddings loader
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
                sentence_encoder = CNNEncoder(
                    token2id=word2id, max_length=self.hyper_params.max_sentence_length, word_size=self.embeddings_vector_size, position_size=self.hyper_params.position_embedding_vector_length,
                    hidden_size=230, blank_padding=True, kernel_size=3, padding_size=1,
                    word2vec=word2vec, dropout=self.hyper_params.dropout, is_embedding_vector=self.hyper_params.is_embedding_vector, features=self.features)

                self.model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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
        pass # below needs to be fixed to be more general
        """
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
                sent_id = eer_example.arg0.sentence.docid + '#' + str(eer_example.arg0.sentence.index)
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
        logging.info('Start of Bert mention predicting for single model')
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

        eer_order_transform = [eer_order_transform[i] for i in curr_indices_ldc]
        return all_eer_predictions
        # return (eer_examples_pt_all, predicted_role_all, confidences, None), eer_order_transform       
        """
