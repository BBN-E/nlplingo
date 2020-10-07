from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging
from collections import defaultdict

import numpy as np

from nlplingo.common.utils import F1Score
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.common.serialize_disk import load_from_serialized

logger = logging.getLogger(__name__)


class SpanPairData(object):
    def __init__(self, examples, data, data_list, label):
        """
        :type examples: list[nlplingo.tasks.spanpair.example.SpanPairExample]
        :type data: defaultdict[str, list[numpy.ndarray]]
        :type data_list: list[numpy.ndarray]
        :type label: numpy.ndarray
        """
        self.examples = examples
        self.data = data
        self.data_list = data_list
        self.label = label


def train_spanpair(params, word_embeddings, extractor, serialize_list, k_partitions=None, partition_id=None):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    if serialize_list is None:
        logger.info("Preparing docs")
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

        print('#### Generating Training data')
        train_data = generate_pair_data_feature(extractor, train_docs, 'train')
        """:type: nlplingo.tasks.spanpair.run.SpanPairData"""

        print('#### Generating Dev data')
        dev_data = generate_pair_data_feature(extractor, dev_docs, 'dev')
        """:type: nlplingo.tasks.spanpair.run.SpanPairData"""

        print('#### Generating Test data')
        test_data = generate_pair_data_feature(extractor, test_docs, 'test')
        """:type: nlplingo.tasks.spanpair.run.SpanPairData"""
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        train_data = generate_pair_data_feature_from_serialized(extractor, train_candidates)
        dev_data = generate_pair_data_feature_from_serialized(extractor, dev_candidates)
        test_data = generate_pair_data_feature_from_serialized(extractor, test_candidates)

    model = extractor.extraction_model
    """:type: nlplingo.nn.spanpair_model.SpanPairModel"""
    model.fit_model(train_data.data_list, train_data.label, [], [])     # forgo validation during training epoch, to save compute time

    # Save model data
    if params['save_model']:
        print('==== Saving SpanPair model ====')
        model.save_keras_model(extractor.model_file)

    # ==== dev data scoring ====
    dev_predictions = model.predict(dev_data.data_list)

    f1_dev = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.examples)

    # we can only do the following if we are not sampling from the dev examples
    #dev_tp = calculate_pair_true_positive(dev_docs, 'dev')
    #f1_dev = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, dev_tp)

    for f1 in f1_dev:
        print('Dev F1 score: {}'.format(f1.to_string()))
    with open(params['train.score_file'], 'w') as o:
        for f1 in f1_dev:
            o.write('F1 score: {}\n'.format(f1.to_string()))

    # ==== test data scoring ====
    test_predictions = model.predict(test_data.data_list)

    f1_test = evaluate_f1_binary(test_predictions, test_data.label, test_data.examples)

    for f1 in f1_test:
        print('Test F1 score: {}'.format(f1.to_string()))
    with open(params['test.score_file'], 'w') as o:
        for f1 in f1_test:
            o.write('F1 score: {}\n'.format(f1.to_string()))


def generate_pair_data_feature(extractor, docs, mode):
    """
    :type extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    feature_generator = extractor.feature_generator
    """:type: nlplingo.tasks.spanpair.feature.SpanPairFeatureGenerator"""
    example_generator = extractor.example_generator
    example_generator.train_dev_test_mode = mode
    """:type: nlplingo.tasks.spanpair.generator.SpanPairExampleGenerator"""

    candidates = example_generator.generate(docs)
    feature_generator.populate(candidates)
    """:type: list[nlplingo.tasks.spanpair.example.SpanPairExample]"""

    data = example_generator.examples_to_data_dict(candidates, feature_generator.feature_setting)

    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]
    label = np.asarray(data['label'])

    return SpanPairData(candidates, data, data_list, label)

def generate_pair_data_feature_from_serialized(extractor, candidates):
    feature_generator = extractor.feature_generator
    """:type: nlplingo.tasks.spanpair.feature.SpanPairFeatureGenerator"""
    example_generator = extractor.example_generator

    data = example_generator.examples_to_data_dict(candidates, feature_generator.feature_setting)

    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]
    label = np.asarray(data['label'])

    return SpanPairData(candidates, data, data_list, label)


def evaluate_f1_binary(prediction, label, examples, pred_threshold=0.5, class_label='OVERALL'):
    """
    Given a binary task, we will always assume 0 class index is negative, 1 is positive

    :type examples: list[nlplingo.tasks.spanpair.example.SpanPairExample]
    :type num_true_positive: int
    :type target_types: set[str]    # a set of event types that we want to evaluate on
    :rtype: nlplingo.common.utils.F1Score
    """
    ret = []

    print('In spanpair.run.evaluate_f1_binary: #prediction={} #label={} #examples={}'.format(len(prediction), len(label), len(examples)))

    num_correct = 0
    num_true = 0
    num_predict = 0
    for i in range(len(prediction)):
        if prediction[i] >= pred_threshold:
            num_predict += 1
            if label[i] == 1:
                num_correct += 1

    for i in range(len(label)):
        if label[i] == 1:
            num_true += 1

    ret.append(F1Score(num_correct, num_true, num_predict, class_label))
    return ret

