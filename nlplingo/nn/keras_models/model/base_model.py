
import abc
import logging

from keras.callbacks import EarlyStopping
from keras.models import Model
import numpy as np

from nlplingo.nn.extraction_model import ExtractionModel
from nlplingo.nn.keras_models.common.layer_creator import LayerCreator


logger = logging.getLogger(__name__)


class KerasExtractionModel(ExtractionModel):

    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.eventargument.feature.EventArgumentFeature
        """
        super(KerasExtractionModel, self).__init__(
            params, extractor_params, event_domain, embeddings, hyper_params,
            features)
        self.is_binary = None

    @property
    @abc.abstractmethod
    def none_label_index(self):
        """Must be defined as property in child classes; cannot be set"""
        raise NotImplementedError

    def create_model(self):

        assert self.num_output is not None

        self.layers = LayerCreator(
            self.features.activated_features,
            self.extractor_params,
            self.hyper_params,
            self.event_domain,
            self.num_output,
            self.word_embeddings,
        )

        self.is_binary = self.layers.is_binary

    def compile(self, model_outputs, model_input_dict, metrics=None):

        if metrics is None:
            metrics = []

        model_inputs = [model_input_dict[k] for k in self.features.activated_features]

        self.model = Model(inputs=model_inputs, outputs=model_outputs)
        # historically self.loss = 'categorical_crossentropy' or masked_bce
        self.model.compile(optimizer=self.optimizer,
                           loss=self.layers.loss,
                           metrics=metrics)

    def fit_model(self, train_data_list, train_label, test_data_list, test_label):
        """
        Overrides
        :param train_data_list:
        :param train_label:
        :param test_data_list:
        :param test_label:
        :return:
        """

        if self.is_binary:
            class_weights = self.prepare_class_weight(train_label)
            sample_weight = None
        else:
            sample_weight = self.prepare_sample_weight(train_label)
            class_weights = None

        logger.debug('- train_data_list={}'.format(train_data_list))
        logger.debug('- train_label={}'.format(train_label))

        callbacks = None
        if self.hyper_params.early_stopping:
            callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

        if len(test_data_list) > 0:
            val_data = (test_data_list, test_label)
        else:
            val_data = None

        history = self.model.fit(
            train_data_list,
            train_label,
            sample_weight=sample_weight,
            class_weight=class_weights,
            batch_size=self.hyper_params.batch_size,
            epochs=self.hyper_params.epoch,
            validation_data=val_data,
            callbacks=callbacks
            )

        return history

    def prepare_sample_weight(self, train_label):
        sample_weight = np.ones(train_label.shape[0])
        if train_label.ndim == 1:  # some tasks have 1d (binary) labels
            # [ 1, 1, 0, 1, 0 ]
            train_label_argmax = train_label
        else:  # most tasks have 2d (multilabel) labels
            # [ [ 1, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 0 ], [ 0, 1 ] ]
            train_label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(train_label_argmax):
            if label_index != self.none_label_index:
                sample_weight[i] = self.hyper_params.positive_weight
        return sample_weight

    def prepare_class_weight(self, train_label):
        # CLASS WEIGHT CODE
        if len(train_label.shape) > 1:
            count_per_label = np.sum(train_label, axis=0)
            # count_all_labels = float(np.sum(count_per_label))
            # total_over_count = count_all_labels / count_per_label
        else:  # binary 1d label
            positives = np.sum(train_label)
            negatives = len(train_label) - positives
            count_per_label = np.asarray([negatives, positives])

        # CODE FOR INVERSE LOG FREQUENCY CLASS WEIGHTS
        inverse_log_freqs = 1 / np.log(count_per_label)
        class_weights = {i: weight for i, weight in enumerate(inverse_log_freqs)}

        logger.debug("CLASS WEIGHTS:")
        logger.debug("{}".format(sorted(class_weights.items(), key=lambda x: x[0])))
        logger.debug("CLASS FREQUENCIES:")
        logger.debug(" ".join(["{}: {}".format(i, x) for i, x in enumerate(count_per_label)]))

        return class_weights
