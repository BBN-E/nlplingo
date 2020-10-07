from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from nlplingo.common.scoring import evaluate_entity_relation_f1
from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.tasks.entityrelation.generator import EntityRelationExampleGenerator
from nlplingo.common.serialize_disk import load_from_serialized

logger = logging.getLogger(__name__)


def generate_entity_relation_data_feature(generator, docs, feature_generator):
    candidates = generator.generate(docs)
    feature_generator.populate(candidates)

    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]

    label = np.asarray(data['label'])

    print('#entity_relation-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def generate_entity_relation_data_feature_from_serialized(generator, feature_generator, candidates):
    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]

    label = np.asarray(data['label'])

    print('#entity_relation-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def train_entity_relation_from_file(params, word_embeddings, entity_relation_extractor, serialize_list, k_partitions=None, partition_id=None):
    feature_generator = entity_relation_extractor.feature_generator
    example_generator = entity_relation_extractor.example_generator
    entity_relation_model = entity_relation_extractor.extraction_model

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(entity_relation_model)={}'.format(type(entity_relation_model)))

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)

        for doc in train_docs:
            doc.apply_domain(entity_relation_extractor.domain)
        for doc in test_docs:
            doc.apply_domain(entity_relation_extractor.domain)

        (train_examples, train_data, train_data_list, train_label) = generate_entity_relation_data_feature(
            example_generator,
            train_docs,
            feature_generator
        )
        print(train_label)

        (dev_examples, dev_data, dev_data_list, dev_label) = generate_entity_relation_data_feature(
            example_generator,
            test_docs,
            feature_generator
        )
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = generate_entity_relation_data_feature_from_serialized(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = generate_entity_relation_data_feature_from_serialized(example_generator, feature_generator, dev_candidates)
        (test_examples, test_data, test_data_list, test_label) = generate_entity_relation_data_feature_from_serialized(example_generator, feature_generator, test_candidates)
        test_tuple = (test_examples, test_data, test_data_list, test_label)

    entity_relation_model.fit_model(train_data_list, train_label, dev_data_list, dev_label)

    predictions = entity_relation_model.predict(dev_data_list)

    score, score_breakdown, gold_labels = evaluate_entity_relation_f1(entity_relation_extractor.domain, dev_label, dev_examples, predictions)
    print('entity_relation-score: ' + score.to_string())

    for index, f1_score in score_breakdown.items():
        er = entity_relation_extractor.domain.get_entity_relation_type_from_index(index)
        print('{}\t{}'.format(er, f1_score.to_string()))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            er = entity_relation_extractor.domain.get_entity_relation_type_from_index(index)
            f.write('{}\t{}\n'.format(er, f1_score.to_string()))

    print('==== Saving entity_relation model ====')
    if params['save_model']:
        entity_relation_model.save_keras_model(entity_relation_extractor.model_file)

    if params['test.score_file']:
        if serialize_list is None:
            test_entity_relation_with_gold_mentions(
                params,
                word_embeddings,
                entity_relation_extractor
            )
        else:
            test_entity_relation_with_gold_mentions(
                params,
                word_embeddings,
                entity_relation_extractor,
                test_tuple=test_tuple
            )

def test_entity_relation_with_gold_mentions(
        params,
        word_embeddings,
        entity_relation_extractor,
        scoring_domain=None,
        test_tuple=None
):
    event_domain = entity_relation_extractor.domain

    if test_tuple is None:
        test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
        for doc in test_docs:
            doc.apply_domain(event_domain)

        # TODO: is this based on gold mentions?
        logging.info('Generating entity_relation examples')
        (entity_relation_examples, entity_relation_data, entity_relation_data_list, entity_relation_label) = generate_entity_relation_data_feature(
            entity_relation_extractor.example_generator, test_docs, entity_relation_extractor.feature_generator)
    else:
        entity_relation_examples, entity_relation_data, entity_relation_data_list, entity_relation_label = test_tuple

    for v in entity_relation_data_list:
        print(v)

    # TODO: is this based on gold mentions?
    logging.info('Predicting entity_relation')
    entity_relation_predictions = entity_relation_extractor.extraction_model.predict(entity_relation_data_list)

    score_with_gold_mentions, score_breakdown_with_gold_mentions, gold_labels = evaluate_entity_relation_f1(
        event_domain, entity_relation_label, entity_relation_examples, entity_relation_predictions, scoring_domain)
    print('entity_relation-scores: {}'.format(score_with_gold_mentions.to_string()))

    for index, f1_score in score_breakdown_with_gold_mentions.items():
        er = event_domain.get_entity_relation_type_from_index(index)
        print('entity_relation-scores: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-mentions: {}\n'.format(score_with_gold_mentions.to_string()))
        for index, f1_score in score_breakdown_with_gold_mentions.items():
            er = event_domain.get_entity_relation_type_from_index(index)
            f.write('With gold-mentions: {}\t{}\n'.format(er, f1_score.to_string()))
