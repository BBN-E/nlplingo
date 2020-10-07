from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

global keras_trigger_model

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# class TriggerModel(ExtractionModel):
#     def __init__(self, params, event_domain, embeddings):
#         """
#         :type: params: dict
#         :type event_domain: nlplingo.tasks.event_domain.EventDomain
#         :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
#         """
#         super(TriggerModel, self).__init__(params, event_domain, embeddings)
#
#         hyper_params = params['hyper-parameters']
#         self.positive_weight = hyper_params['positive_weight']
#         self.epoch = hyper_params['epoch']
#         self.early_stopping = hyper_params.get('early_stopping', False)
#         self.number_of_feature_maps = hyper_params.get('number_of_feature_maps', 0)  # number of convolution feature maps
#         self.batch_size = hyper_params['batch_size']
#
#         self.position_embedding_vector_length = hyper_params['position_embedding_vector_length']
#         self.cnn_filter_lengths = hyper_params.get('cnn_filter_lengths', 0)
#         """:type: list[int]"""
#         self.dropout = hyper_params['dropout']
#
#         self.entity_embedding_vector_length = hyper_params['entity_embedding_vector_length']
#         self.use_bio_index = params['model_flags'].get('use_bio_index', False)
#         self.use_lex_info = params['model_flags'].get('use_lex_info', False)
#
#         self.number_of_entity_bio_types = len(event_domain.entity_bio_types)
#
#         self.num_output = len(event_domain.event_types)
#         self.model = None
#
#     def fit(self, train_data_list, train_label, test_data_list, test_label):
#         global keras_trigger_model
#
#         if self.verbosity == 1:
#             print('- train_data_list=', train_data_list)
#             print('- train_label=', train_label)
#
#         none_label_index = self.event_domain.get_event_type_index('None')
#         sample_weight = np.ones(train_label.shape[0])
#         label_argmax = np.argmax(train_label, axis=1)
#         for i, label_index in enumerate(label_argmax):
#             if label_index != none_label_index:
#                 sample_weight[i] = self.positive_weight
#
#         callbacks = None
#         if self.early_stopping:
#             callbacks = [early_stopping]
#
#         if len(test_label) > 0:
#             history = keras_trigger_model.fit(
#                 train_data_list,
#                 train_label,
#                 sample_weight=sample_weight,
#                 batch_size=self.batch_size,
#                 nb_epoch=self.epoch,
#                 validation_data=(
#                     test_data_list,
#                     test_label
#                 ),
#                 callbacks=callbacks
#             )
#
#         else:
#             history = keras_trigger_model.fit(
#                 train_data_list,
#                 train_label,
#                 sample_weight=sample_weight,
#                 batch_size=self.batch_size,
#                 nb_epoch=self.epoch,
#                 callbacks=callbacks
#             )
#         return history
#
#     # def fit(self, train_data_list, train_label, test_data_list, test_label):
#     #     if self.verbosity == 1:
#     #         print('- train_data_list=', train_data_list)
#     #         print('- train_label=', train_label)
#     #
#     #     none_label_index = self.event_domain.get_event_type_index('None')
#     #     sample_weight = np.ones(train_label.shape[0])
#     #     label_argmax = np.argmax(train_label, axis=1)
#     #     for i, label_index in enumerate(label_argmax):
#     #         if label_index != none_label_index:
#     #             sample_weight[i] = self.positive_weight
#     #
#     #     if len(test_label) > 0:
#     #         history = self.model.fit(train_data_list, train_label,
#     #                               sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch,
#     #                               validation_data=(test_data_list, test_label), callbacks=[early_stopping])
#     #     else:
#     #         history = self.model.fit(train_data_list, train_label,
#     #                                           sample_weight=sample_weight,
#     #                                           batch_size=self.batch_size, nb_epoch=self.epoch, callbacks=[early_stopping])
#     #     return history
#
#
#     def load_keras_model(self, filename=None):
#         global keras_trigger_model
#         keras_trigger_model = keras.models.load_model(filename, self.keras_custom_objects)
#
#     def save_keras_model(self, filename):
#         global keras_trigger_model
#         keras_trigger_model.save(filename)
#         print(keras_trigger_model.summary())
#
#     def predict(self, test_data_list):
#         global keras_trigger_model
#
#         try:
#             pred_result = keras_trigger_model.predict(test_data_list)
#         except:
#             self.load_keras_model(filename=os.path.join(self.model_dir, 'eventtrigger.hdf'))
#             print('*** Loaded keras_trigger_model ***')
#             pred_result = keras_trigger_model.predict(test_data_list)
#         return pred_result



