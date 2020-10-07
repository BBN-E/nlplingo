from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os

import keras
import numpy as np
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from nlplingo.sandbox.model import EventExtractionModel

global keras_domain_model


class DomainModel(EventExtractionModel):
    def __init__(self, params, embeddings, event_domain):
        """
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(DomainModel, self).__init__(params, event_domain, embeddings,
                                           batch_size=params.get_int('domain.batch_size'),
                                           num_feature_maps=params.get_int('domain.num_feature_maps'))
        self.num_output = len(event_domain.event_types)
        self.positive_weight = params.get_float('domain.positive_weight')
        self.epoch = params.get_int('domain.epoch')

    def fit(self, train_data_list, train_label):
        global keras_domain_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = 0
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = train_label
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight

        history = keras_domain_model.fit(train_data_list, train_label,
                                  sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch)
        return history

    def load_keras_model(self, filename=None):
        global keras_domain_model
        keras_domain_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_domain_model
        keras_domain_model.save(filename)
        print(keras_domain_model.summary())

    def predict(self, test_data_list):
        global keras_domain_model

        try:
            pred_result = keras_domain_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'domain.hdf'))
            print('*** Loaded keras_domain_model ***')
            pred_result = keras_domain_model.predict(test_data_list)
        return pred_result

class MaxPoolEmbeddedDomainModel(DomainModel):
    def __init__(self, params, embeddings, event_domain):
        """
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedDomainModel, self).__init__(params, embeddings, event_domain)
        self.train_embedding = False
        self.create_model()

    def create_model(self):
        global keras_domain_model

        context_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'word_vector')
        all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                  weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')(all_words)
        maxpool = GlobalMaxPooling1D()(conv)

        dropout = Dropout(self.dropout)(maxpool)
        out = Dense(self.num_output, activation=u'softmax')(dropout)

        keras_domain_model = Model(input=[context_input], output=[out])

        keras_domain_model.compile(optimizer=self.optimizer, loss=u'categorical_crossentropy', metrics=[])


