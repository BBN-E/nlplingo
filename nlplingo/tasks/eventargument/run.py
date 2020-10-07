from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import numpy as np

from keras.models import load_model as keras_load_model

from nlplingo.common.scoring import evaluate_arg_f1
from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.tasks.eventargument.generator import EventArgumentExampleGenerator

from nlplingo.tasks.eventtrigger.run import generate_trigger_data_feature
from nlplingo.tasks.eventtrigger.run import get_predicted_positive_triggers

import glob

from nlplingo.common.serialize_disk import ChunkWriter, WRITE_THRESHOLD, ensure_dir, load_disk_dirs, get_test_examples_labels
from nlplingo.tasks.sequence.generator import BIOGenerator
from nlplingo.tasks.sequence.run import train_argument_from_docs

from nlplingo.common.serialize_disk import load_from_serialized

logger = logging.getLogger(__name__)


def load_argument_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'argument.hdf'))
    return model

def load_argument_modelfile(filepath):
    return keras_load_model(str(filepath))

def generate_argument_data_feature(generator, docs, feature_generator, predicted_triggers=None):
    """
    +1
    :type generator: nlplingo.tasks.eventargument.generator.EventArgumentExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator
    :type predicted_triggers: defaultdict(list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint])
    """
    candidates = generator.generate(docs, triggers=predicted_triggers)
    feature_generator.populate(candidates)
    """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""

    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]

    label = np.asarray(data['label'])

    print('#arg-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def generate_argument_data_feature_from_serialized(generator, feature_generator, candidates):
    """
    +1
    :type generator: nlplingo.tasks.eventargument.generator.EventArgumentExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator
    :type predicted_triggers: defaultdict(list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint])
    """
    data = generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]

    label = np.asarray(data['label'])

    print('#arg-examples=%d' % (len(candidates)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (candidates, data, data_list, label)

def train_argument(params, word_embeddings, argument_extractor, serialize_list, k_partitions=None, partition_id=None):
    """
    :type params: dict
    :type word_embeddings: dict[str:nlplingo.embeddings.word_embeddings.WordEmbedding]
    :type argument_extractor: nlplingo.nn.extraactor.Extractor
    """

    #if argument_extractor.engine == 'transformers':
    if argument_extractor.model_type.startswith('sequence_'):
        return train_argument_from_docs(params, argument_extractor)
        #return train_transformer_sequence_argument(params, argument_extractor)
    elif argument_extractor.model_type.startswith('oregon'):
        from nlplingo.oregon.nlplingo.tasks.sequence.run import train_argument_from_docs as oregon_train_argument_from_docs
        return oregon_train_argument_from_docs(params, argument_extractor.extractor_params, argument_extractor.domain, argument_extractor.hyper_parameters)

    feature_generator = argument_extractor.feature_generator
    example_generator = argument_extractor.example_generator
    argument_model = argument_extractor.extraction_model
    """:type: nlplingo.nn.argument_model.ArgumentModel"""

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(argument_model)={}'.format(type(argument_model)))

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)

        for doc in train_docs:
           doc.apply_domain(argument_extractor.domain)
        for doc in dev_docs:
           doc.apply_domain(argument_extractor.domain)
        (train_examples, train_data, train_data_list, train_label) = generate_argument_data_feature(
            example_generator,
            train_docs,
            feature_generator
        )
        print(train_label)

        (dev_examples, dev_data, dev_data_list, dev_label) = generate_argument_data_feature(
            example_generator,
            dev_docs,
            feature_generator
        )
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = generate_argument_data_feature_from_serialized(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = generate_argument_data_feature_from_serialized(example_generator, feature_generator, dev_candidates)
        (test_examples, test_data, test_data_list, test_label) = generate_argument_data_feature_from_serialized(example_generator, feature_generator, test_candidates)
        test_tuple = (test_examples, test_data, test_data_list, test_label)

    argument_model.fit_model(train_data_list, train_label, dev_data_list, dev_label)
    predictions = argument_model.predict(dev_data_list)


    score, score_breakdown, gold_labels = evaluate_arg_f1(argument_extractor.domain, dev_label, dev_examples, predictions)
    print('Arg-score: ' + score.to_string())

    for index, f1_score in score_breakdown.items():
        er = argument_extractor.domain.get_event_role_from_index(index)
        print('{}\t{}'.format(er, f1_score.to_string()))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            er = argument_extractor.domain.get_event_role_from_index(index)
            f.write('{}\t{}\n'.format(er, f1_score.to_string()))

    print('==== Saving Argument model ====')
    if params['save_model'] and (('engine' not in argument_extractor.extractor_params) or (argument_extractor.extractor_params['engine'] == 'keras')):
        argument_model.save_keras_model(argument_extractor.model_file)
    else:
        argument_model.trained_model.save(argument_model.hyper_params.dict['save_dir'] + '/final_model.pt')

    if params.get('test.score_file'):
        if serialize_list is None:
            test_argument_with_gold_triggers(
                params,
                word_embeddings,
                argument_extractor
            )
        else:
            test_argument_with_gold_triggers(
                params,
                word_embeddings,
                argument_extractor,
                test_tuple=test_tuple
            )

def test_argument_with_gold_triggers(
        params,
        word_embeddings,
        argument_extractor,
        scoring_domain=None,
        test_tuple=None
):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbeddingsAbstract
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type event_domain: nlplingo.tasks.event_domain.EventDomain
    :type scoring_domain: nlplingo.tasks.event_domain.EventDomain
    """
    event_domain = argument_extractor.domain

    if test_tuple is None:
        test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
        for doc in test_docs:
            doc.apply_domain(event_domain)

        logging.info('Generating argument examples based on gold triggers')
        (arg_examples, arg_data, arg_data_list, arg_label) = generate_argument_data_feature(
            argument_extractor.example_generator, test_docs, argument_extractor.feature_generator)
    else:
        arg_examples, arg_data, arg_data_list, arg_label = test_tuple

    for v in arg_data_list:
        print(v)

    logging.info('Predicting arguments based on gold triggers')
    argument_predictions = argument_extractor.extraction_model.predict(arg_data_list)

    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_arg_f1(
        event_domain, arg_label, arg_examples, argument_predictions, scoring_domain)
    print('Arg-scores with gold-triggers: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with gold-triggers: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-triggers: {}\n'.format(score_with_gold_triggers.to_string()))
        for index, f1_score in score_breakdown_with_gold_triggers.items():
            er = event_domain.get_event_role_from_index(index)
            f.write('With gold-triggers: {}\t{}\n'.format(er, f1_score.to_string()))


def test_argument(params, word_embeddings, trigger_extractor, argument_extractor, scoring_domain=None):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbeddingsAbstract
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type scoring_domain: nlplingo.tasks.event_domain.EventDomain
    """
    event_domain = argument_extractor.domain

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
    for doc in test_docs:
        doc.apply_domain(event_domain)

    logging.info('Generating trigger examples')
    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_extractor.example_generator, test_docs, trigger_extractor.feature_generator)

    logging.info('Generating argument examples based on gold triggers')
    (arg_examples, arg_data, arg_data_list, arg_label) = generate_argument_data_feature(
        argument_extractor.example_generator, test_docs, argument_extractor.feature_generator)

    for v in arg_data_list:
        print(v)

    logging.info('Predicting triggers')
    trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples, trigger_extractor)

    logging.info('Predicting arguments based on gold triggers')
    argument_predictions = argument_extractor.extraction_model.predict(arg_data_list)

    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_arg_f1(
        event_domain, arg_label, arg_examples, argument_predictions, scoring_domain)
    print('Arg-scores with gold-triggers: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with gold-triggers: {}\t{}'.format(er, f1_score.to_string()))

    logging.info('Generating argument examples based on predicted triggers')
    (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
        generate_argument_data_feature(
            argument_extractor.example_generator, test_docs, argument_extractor.feature_generator,
            predicted_triggers=predicted_positive_triggers)

    logging.info('Predicting arguments based on predicted triggers')
    argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)

    # evaluate arguments with predicted triggers
    score_with_predicted_triggers, score_breakdown_with_predicted_triggers, pred_labels = \
        evaluate_arg_f1(event_domain, arg_label_pt, arg_examples_pt, argument_predictions_pt, scoring_domain, gold_labels=gold_labels)

    print('Arg-scores with predicted-triggers: {}'.format(score_with_predicted_triggers.to_string()))

    for index, f1_score in score_breakdown_with_predicted_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with predicted-triggers: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-triggers: {}\n'.format(score_with_gold_triggers.to_string()))
        for index, f1_score in score_breakdown_with_gold_triggers.items():
            er = event_domain.get_event_role_from_index(index)
            f.write('With gold-triggers: {}\t{}\n'.format(er, f1_score.to_string()))
        f.write('With-predicted-triggers: {}\n'.format(score_with_predicted_triggers.to_string()))
        for index, f1_score in score_breakdown_with_predicted_triggers.items():
            er = event_domain.get_event_role_from_index(index)
            f.write('With predicted-triggers: {}\t{}\n'.format(er, f1_score.to_string()))



def train_transformer_sequence_argument(params, argument_extractor):
    """
    :type params: dict
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    """
    bp_filepath = params['bp_file']
    keep_unannotated_sentences = argument_extractor.extractor_params['keep_unannotated_sentences']
    tokenization_type = argument_extractor.extractor_params['tokenization_type']     # SPACE, SERIF

    labels = []
    labels.extend('B-{}'.format(label) for label in sorted(argument_extractor.domain.event_roles.keys()) if label != 'None')
    labels.extend('I-{}'.format(label) for label in sorted(argument_extractor.domain.event_roles.keys()) if label != 'None')
    labels.append('O')

    bio_generator = BIOGenerator()

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: train_docs: list[nlplingo.text.text_theory.Document]"""

    bio_lines = bio_generator.generate_argument_bio(bp_filepath, keep_unannotated_sentences, tokenization_type, train_docs)

    train_transformer_sequence(params, argument_extractor.extractor_params, bio_lines, labels)
