from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from nlplingo.common.scoring import evaluate_eer_f1
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.eventrelation.postprocess import prefilter
from nlplingo.tasks.eventrelation.mention_pool_filter import mention_pool_filter_from_memory
from nlplingo.common.serialize_disk import load_from_serialized
from nlplingo.tasks.cross_task_run import generate_binary_triplets_with_relations_from_candidates
import json, codecs
import os

logger = logging.getLogger(__name__)

def generate_eer_data_feature(generator, docs, feature_generator):
    """
    Generate eer data features over nlplingo documents.
    :param generator:  nlplingo.tasks.common.generator.ExampleGenerator
    :param docs: list[nlplingo.text.text_theory.Document]
    :param feature_generator: nlplingo.tasks.common.generator.FeatureGenerator
    :return:
    """
    candidates = generator.generate(docs)
    feature_generator.populate(candidates)

    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = []
    for k in feature_generator.feature_setting.activated_features:
        data_list.append(np.asarray(data[k]))

    label = np.asarray(data['label'])

    print('#eventrelation-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def generate_eer_data_feature_from_serialized(generator, feature_generator, candidates):
    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = []
    for k in feature_generator.feature_setting.activated_features:
        data_list.append(np.asarray(data[k]))

    label = np.asarray(data['label'])

    print('#eventrelation-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def generate_opennre_triplets_from_candidates(candidates):
    triplets = []
    for candidate in candidates:
        triplets.append(candidate.to_triplet())
    return triplets

def generate_eer_data_feature_opennre(generator, docs):
    """
    Generate eer data features for opennre decoding.
    :param generator:  nlplingo.tasks.common.generator.ExampleGenerator
    :param docs: list[nlplingo.text.text_theory.Document]
    :param feature_generator: nlplingo.tasks.common.generator.FeatureGenerator
    :return:
    """
    candidates, decode_triplets = generator.generate_with_triplets(docs)
    logging.info('#eventrelation-examples=%s', len(candidates))
    return (candidates, decode_triplets)


def train_eer_from_json(params, eer_extractor):
    train_path = params['data']['train']['txt']
    dev_path = params['data']['dev']['txt']
    eer_model = eer_extractor.extraction_model

    logger.debug('type(eer_model)={}'.format(type(eer_model)))
    eer_model.fit_txt(train_path, dev_path, None)

def train_eer_from_json_and_test(params, eer_extractor):
    train_path = params['data']['train']['txt']
    dev_path = params['data']['dev']['txt']
    test_path = params['data']['test']['txt']
    eer_model = eer_extractor.extraction_model

    logger.debug('type(eer_model)={}'.format(type(eer_model)))
    eer_model.fit_txt(train_path, dev_path, test_path)

def train_eer_from_file(params, word_embeddings, eer_extractor, serialize_list, k_partitions=None, partition_id=None):
    feature_generator = eer_extractor.feature_generator
    example_generator = eer_extractor.example_generator
    eer_model = eer_extractor.extraction_model

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(eer_model)={}'.format(type(eer_model)))

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)

        for doc in train_docs:
            doc.apply_domain(eer_extractor.domain)
        for doc in test_docs:
            doc.apply_domain(eer_extractor.domain)

        (train_examples, train_data, train_data_list, train_label) = generate_eer_data_feature(
            example_generator,
            train_docs,
            feature_generator
        )
        print(train_label)

        (dev_examples, dev_data, dev_data_list, dev_label) = generate_eer_data_feature(
            example_generator,
            test_docs,
            feature_generator
        )
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, dev_candidates)
        (test_examples, test_data, test_data_list, test_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, test_candidates)
        test_tuple = (test_examples, test_data, test_data_list, test_label)

    eer_model.fit_model(train_data_list, train_label, dev_data_list, dev_label)

    predictions = eer_model.predict(dev_data_list)

    score, score_breakdown, gold_labels = evaluate_eer_f1(eer_extractor.domain, dev_label, dev_examples, predictions)
    print('EER-score: ' + score.to_string())

    for index, f1_score in score_breakdown.items():
        er = eer_extractor.domain.get_eer_type_from_index(index)
        print('{}\t{}'.format(er, f1_score.to_string()))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            er = eer_extractor.domain.get_eer_type_from_index(index)
            f.write('{}\t{}\n'.format(er, f1_score.to_string()))

    print('==== Saving EER model ====')
    if params['save_model']:
        eer_model.save_keras_model(eer_extractor.model_file)

    if params['test.score_file']:
        if serialize_list is None:
            test_eer_with_gold_events(
                params,
                word_embeddings,
                eer_extractor
            )
        else:
            test_eer_with_gold_events(
                params,
                word_embeddings,
                eer_extractor,
                test_tuple=test_tuple
            )

def train_eer_from_file_pytorch(params, word_embeddings, eer_extractor, serialize_list, k_partitions=None, partition_id=None):
    feature_generator = eer_extractor.feature_generator
    example_generator = eer_extractor.example_generator
    eer_model = eer_extractor.extraction_model

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(eer_model)={}'.format(type(eer_model)))

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)

        for doc in train_docs:
            doc.apply_domain(eer_extractor.domain)
        for doc in test_docs:
            doc.apply_domain(eer_extractor.domain)

        (train_examples, train_data, train_data_list, train_label) = generate_eer_data_feature(
            example_generator,
            train_docs,
            feature_generator
        )
        print(train_label)

        (dev_examples, dev_data, dev_data_list, dev_label) = generate_eer_data_feature(
            example_generator,
            test_docs,
            feature_generator
        )
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, dev_candidates)
        (test_examples, test_data, test_data_list, test_label) = generate_eer_data_feature_from_serialized(example_generator, feature_generator, test_candidates)
        # test_tuple = (test_examples, test_data, test_data_list, test_label)

    if eer_model.model_type == 'event-event-relation_cnn-embedded' and eer_model.hyper_params.encoder == 'bert_mention':
        train_triplets = generate_binary_triplets_with_relations_from_candidates(train_examples)
        dev_triplets = generate_binary_triplets_with_relations_from_candidates(dev_examples)
        interm_dir = eer_model.hyper_params.save_model_path + '/interm'
        if not os.path.isdir(interm_dir):
            os.makedirs(interm_dir)
        train_path = interm_dir + '/train_mention_pool.json'
        dev_path = interm_dir + '/dev_mention_pool.json'
        mention_pool_filter_from_memory(train_triplets, train_path)
        mention_pool_filter_from_memory(dev_triplets, dev_path)
        eer_model.fit_txt(train_path, dev_path, None)
    else:
        raise Exception('No other models supported currently.')


def test_eer_with_gold_events(
        params,
        word_embeddings,
        eer_extractor,
        scoring_domain=None,
        test_tuple=None
):
    event_domain = eer_extractor.domain

    if test_tuple is None:
        test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
        for doc in test_docs:
            doc.apply_domain(event_domain)

        # TODO: is this based on gold triggers?
        logging.info('Generating EER examples')
        (eer_examples, eer_data, eer_data_list, eer_label) = generate_eer_data_feature(
            eer_extractor.example_generator, test_docs, eer_extractor.feature_generator)
    else:
        eer_examples, eer_data, eer_data_list, eer_label = test_tuple

    for v in eer_data_list:
        print(v)

    # TODO: is this based on gold triggers?
    logging.info('Predicting EER')
    eer_predictions = eer_extractor.extraction_model.predict(eer_data_list)

    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_eer_f1(
        event_domain, eer_label, eer_examples, eer_predictions, scoring_domain)
    print('EER-scores: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_eer_type_from_index(index)
        print('EER-scores: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-triggers: {}\n'.format(score_with_gold_triggers.to_string()))
        for index, f1_score in score_breakdown_with_gold_triggers.items():
            er = event_domain.get_eer_type_from_index(index)
            f.write('With gold-triggers: {}\t{}\n'.format(er, f1_score.to_string()))


def test_eer_with_gold_events_opennre(
        params,
        word_embeddings,
        eer_extractors,
        scoring_domain=None
):
    event_domain = eer_extractors[0].domain
    none_class_index = event_domain.get_eer_type_index('NA')

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
    for doc in test_docs:
        doc.apply_domain(event_domain)

    # TODO: is this based on gold triggers?
    logging.info('Generating EER examples to decode.')
    eer_examples, _, _, _ = generate_eer_data_feature(eer_extractors[0].example_generator, test_docs,
                                                      eer_extractors[0].feature_generator)
    decode_triplets = generate_opennre_triplets_from_candidates(eer_examples)
    #for v in eer_data_list:
    #    print(v)

    # TODO: is this based on gold triggers?
    logging.info('Predicting EER')

    # prefilter examples
    original_indices = list(range(len(eer_examples)))
    prefilter(original_indices, eer_examples)
    eer_examples = [eer_examples[i] for i in original_indices]
    decode_triplets = [decode_triplets[i] for i in original_indices]

    eer_examples_pt_all, predicted_role_all, confidences = eer_extractors[0].extraction_model.predict_triplets(eer_examples,
                                                                                                           decode_triplets,
                                                                                                           event_domain,
                                                                                                           none_class_index)

    """
    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_eer_f1(
        event_domain, eer_label, eer_examples, eer_predictions, scoring_domain)
    print('EER-scores: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_eer_type_from_index(index)
        print('EER-scores: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-triggers: {}\n'.format(score_with_gold_triggers.to_string()))
        for index, f1_score in score_breakdown_with_gold_triggers.items():
            er = event_domain.get_eer_type_from_index(index)
            f.write('With gold-triggers: {}\t{}\n'.format(er, f1_score.to_string()))
    """