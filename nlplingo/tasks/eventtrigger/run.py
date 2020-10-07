from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import logging
import random
import codecs
import copy

from collections import defaultdict

import numpy as np
import glob

from keras.models import load_model as keras_load_model

from nlplingo.common.scoring import evaluate_f1, calculate_confusion_matrix, print_confusion_matrix_for_event_types
from nlplingo.common.scoring import evaluate_multi_label_f1
from nlplingo.common.scoring import evaluate_baseline_and_best_multi_label_f1s
from nlplingo.common.scoring import print_score_breakdown
from nlplingo.common.scoring import write_score_to_file
from nlplingo.common.scoring import report_top_n_sample_info
from nlplingo.common.scoring import calculate_all_thresholds
from nlplingo.common.utils import F1Score
from nlplingo.active_learning.active_batch_selector import ActiveBatchSelectorFactory

from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator


from nlplingo.tasks.eventtrigger.generator import EventKeywordList
from nlplingo.tasks.eventtrigger.metric import get_recall_misses as get_trigger_recall_misses
from nlplingo.tasks.eventtrigger.metric import get_precision_misses as get_trigger_precision_misses

from nlplingo.common.serialize_disk import divide_chunks
from nlplingo.common.serialize_disk import ChunkWriter, WRITE_THRESHOLD, ensure_dir, load_disk_dirs, get_test_examples_labels
from nlplingo.tasks.sequence.generator import BIOGenerator
from nlplingo.tasks.sequence.run import train_trigger_from_docs
from nlplingo.common.serialize_disk import load_from_serialized

logger = logging.getLogger(__name__)


# +1
def get_predicted_positive_triggers(predictions, examples, extractor):
    """Collect the predicted positive triggers and organize them by docid
    Also, use the predicted event_type for each such trigger example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    event_domain = extractor.domain
    """:type event_domain: nlplingo.tasks.event_domain.EventDomain"""
    none_class_index = event_domain.get_event_type_index('None')
    assert len(predictions) == len(examples)
    ret = defaultdict(list)

    if extractor.extraction_model.is_binary:

        # multi-label models use thresholds to find positive triggers
        thresholds = extractor.class_thresholds

        for i, labels in enumerate(predictions):    # for each example
            for j, score in enumerate(labels):      # for each class prediction of the example
                score = predictions[i, j]
                if j != none_class_index and score >= thresholds[j]:
                    # Copying required so changing `event_type` is possible
                    eg = copy.copy(examples[i])
                    """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
                    eg.anchor = copy.copy(examples[i].anchor)
                    eg.event_type = event_domain.get_event_type_from_index(j)
                    eg.anchor.label = eg.event_type
                    eg.score = score
                    ret[eg.sentence.docid].append(eg)

    else:  # single-label models use argmax to find positive triggers
        pred_arg_max = np.argmax(predictions, axis=1)
        for i, index in enumerate(pred_arg_max):
            if index != none_class_index:
                eg = examples[i]
                """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
                eg.event_type = event_domain.get_event_type_from_index(index)
                eg.anchor.label = eg.event_type
                eg.score = predictions[i][index]
                ret[eg.sentence.docid].append(eg)
    return ret


def get_predicted_positive_triggers_with_active_learning_metric(
        predictions,
        examples,
        none_class_index,
        event_domain,
        active_learning_metric
):
    """Collect the predicted positive triggers and organize them by docid
    Also, use the predicted event_type for each such trigger example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
    :type event_domain: nlplingo.tasks.event_domain.EventDomain
    """
    assert len(predictions) == len(examples)
    ret = defaultdict(list)

    pred_arg_max = np.argmax(predictions, axis=1)
    active_learning_metric_result = active_learning_metric(predictions)
    for i, (index, active_learning_raw_score) in enumerate(zip(pred_arg_max, active_learning_metric_result)):
        if index != none_class_index:
            # print(predictions[i][index])
            eg = examples[i]
            """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            eg.anchor.label = eg.event_type
            eg.score = predictions[i][index]
            eg.active_learning_raw_score = active_learning_raw_score
            ret[eg.sentence.docid].append(eg)
    return ret


def check_polysemy(examples):

    types_and_triggers = defaultdict(lambda: defaultdict(int))
    for eg in examples:
        types_and_triggers[eg.event_type][eg.anchor.head().text.lower()] += 1

    polysemous_triggers = defaultdict(set)
    polysemous_positive_triggers = defaultdict(set)
    for type_a, triggers in types_and_triggers.items():
        for type_b in types_and_triggers.keys():
            if type_a == type_b:
                continue
            for trigger in triggers:
                if (types_and_triggers[type_a][trigger] > 0
                        and types_and_triggers[type_b][trigger] > 0):
                    polysemous_triggers[trigger].add(type_a)
                    polysemous_triggers[trigger].add(type_b)
                    if type_a != 'None' and type_b != 'None':
                        polysemous_positive_triggers[trigger].add(type_a)
                        polysemous_positive_triggers[trigger].add(type_b)

    monosemous_coverage = defaultdict(int)
    monosemous_positive_coverage = defaultdict(int)
    for type_a, triggers in types_and_triggers.items():
        for trigger in triggers:
            if trigger not in polysemous_triggers:
                monosemous_coverage[type_a] += triggers[trigger]
            if trigger not in polysemous_positive_triggers:
                monosemous_positive_coverage[type_a] += triggers[trigger]
        total = sum(types_and_triggers[type_a].values())
        logger.info(
            "Class `{}` triggers are {}% monosemous (including NA samples) "
            "({}/{})".format(
                type_a,
                100 * monosemous_coverage[type_a] / float(total),
                monosemous_coverage[type_a],
                total))
        logger.info(
            " - {}% monosemous (ignoring NA samples) ({}/{})".format(
                100 * monosemous_positive_coverage[type_a] / float(total),
                monosemous_positive_coverage[type_a],
                total))

    for trigger, types in polysemous_triggers.items():
        logger.info("Trigger `{}` is polysemous (including NA samples) for "
                    "classes:".format(trigger))
        total_for_trigger = sum(types_and_triggers[t][trigger] for t in types)
        for type_a in types:
            total_for_type = sum(types_and_triggers[type_a].values())
            logger.info(
                " - {}% ({}/{}) of class `{}` ({}% ({}/{}) of trigger's "
                "appearances)".format(
                    (100 * types_and_triggers[type_a][trigger]
                     / float(total_for_type)),
                    types_and_triggers[type_a][trigger],
                    total_for_type,
                    type_a,
                    (100 * types_and_triggers[type_a][trigger]
                     / float(total_for_trigger)),
                    types_and_triggers[type_a][trigger],
                    total_for_trigger
                ))

    for trigger, types in polysemous_positive_triggers.items():
        logger.info("Trigger `{}` is polysemous (ignoring NA samples) for "
                    "classes:".format(trigger))
        total_for_trigger = sum(types_and_triggers[t][trigger]
                                for t in types if t != 'None')
        for type_a in types:
            total_for_type = sum(types_and_triggers[type_a].values())
            logger.info(
                " - {}% ({}/{}) of class `{}` ({}% ({}/{}) of trigger's "
                "positive appearances)".format(
                    (100 * types_and_triggers[type_a][trigger]
                     / float(total_for_type)),
                    types_and_triggers[type_a][trigger],
                    total_for_type,
                    type_a,
                    (100 * types_and_triggers[type_a][trigger]
                     / float(total_for_trigger)),
                    types_and_triggers[type_a][trigger],
                    total_for_trigger
                ))


def generate_trigger_data_feature(example_generator, docs, feature_generator):
    """
    :type example_generator: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
    """

    logger.info("GENERATING EXAMPLES")
    candidates = example_generator.generate(docs)
    feature_generator.populate(candidates)
    """:type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""

    data = example_generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]
    label = np.asarray(data['label'])

    # logger.info("Checking polysemy of trainset:")
    # check_polysemy(candidates)

    return (candidates, data, data_list, label)

def generate_trigger_data_feature_from_serialized(example_generator, feature_generator, candidates):
    """
    :type example_generator: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
    """
    data = example_generator.examples_to_data_dict(candidates, feature_generator.feature_setting)
    data_list = [np.asarray(data[k]) for k in feature_generator.feature_setting.activated_features]
    label = np.asarray(data['label'])

    # logger.info("Checking polysemy of trainset:")
    # check_polysemy(candidates)

    return (candidates, data, data_list, label)

# +1
def apply_positive_training_trigger_filter_to_predictions(examples, preds, extractor):
    """Modify predictions to account for non-positive trigger filtration"""
    logger.info("Applying Positive Training Trigger filter to predictions:")

    all_NAs = 0
    kept_NAs = 0
    all_pos = 0
    kept_pos = 0
    canceled_predictions = 0
    samples_with_canceled_predictions = set()
    keywords = extractor.example_generator.positive_triggers

    if not keywords:  # function should not be called with empty or null kws
        raise IOError("No keywords supplied to prediction keyword filter "
                      "(keywords={})".format(repr(keywords)))

    for i, ex in enumerate(examples):
        trigger = ex.get_keyword()

        # version of function for allowing any type:
        '''
        if not any([trigger in t for t in keywords.values()]):
            preds[i] = 0.0
            preds[i, none_idx] = 1.0
        '''

        # version of function for allowing only type-specific keywords:
        for label_idx in range(preds.shape[1]):
            label = extractor.domain.get_event_type_from_index(label_idx)
            unseen_trigger = trigger not in keywords.get(label, {})

            if unseen_trigger:  # we have not seen this trigger-word in the positive examples of this class
                preds[i, label_idx] = 0.0
                canceled_predictions += 1
                samples_with_canceled_predictions.add(i)

        # report numbers
        if ex.event_type == 'None' or ex.event_type == {'None'}:
            all_NAs += 1
            if any([trigger in t for t in keywords.values()]):
                kept_NAs += 1
        else:
            if isinstance(ex.event_type, str):
                event_types = {ex.event_type}
            else:
                event_types = ex.event_type
            all_pos += len(event_types)
            if any([trigger in t for t in keywords.values()]):
                kept_pos += len(event_types)

    logger.info("POSITIVE TRIGGERS SEEN FOR {} of {} NA INSTANCES".format(
        kept_NAs, all_NAs))
    logger.info("POSITIVE TRIGGERS SEEN FOR {} of {} POSITIVE INSTANCES".format(
        kept_pos, all_pos))
    logger.info("Zeroed out {} predictions in {} samples".format(
        canceled_predictions, len(samples_with_canceled_predictions)))

# +1
def train_trigger_from_file(params, word_embeddings, trigger_extractor, serialize_list, k_partitions=None, partition_id=None):
    """
    :type params: dict
    :type word_embeddings: dict[str:nlplingo.embeddings.word_embeddings.WordEmbedding]
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    """

    #if trigger_extractor.engine == 'transformers':
    if trigger_extractor.model_type.startswith('sequence_'):
        #return train_transformer_sequence_trigger(params, trigger_extractor)
        return train_trigger_from_docs(params, trigger_extractor)
    elif trigger_extractor.model_type.startswith('oregon'):
        from nlplingo.oregon.nlplingo.tasks.sequence.run import train_trigger_from_docs as oregon_train_trigger_from_docs
        return oregon_train_trigger_from_docs(params, trigger_extractor.extractor_params, trigger_extractor.domain, trigger_extractor.hyper_parameters)

    # prepare objects/paths needed for fit
    if not os.path.isdir(os.path.dirname(trigger_extractor.model_file)):
        raise IOError("No such directory {}"
                      .format(os.path.dirname(trigger_extractor.model_file)))

    feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator"""
    example_generator = trigger_extractor.example_generator
    """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""
    trigger_model = trigger_extractor.extraction_model
    """:type: nlplingo.nn.trigger_model.TriggerModel"""

    # Determine method for scoring/predicting this classifier
    classifier_is_binary = trigger_model.is_binary

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(trigger_model)={}'.format(type(trigger_model)))

    if serialize_list is None:
        # prepare dataset for sample generation
        logger.info("Preparing docs")
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
        logger.info("Applying domain")
        for doc in train_docs + test_docs:
            doc.apply_domain(trigger_extractor.domain)

        # generate dataset samples for fit
        logger.info("GENERATING TRAINING EXAMPLES")
        (train_examples, train_data, train_data_list, train_label) = (
            generate_trigger_data_feature(
                example_generator, train_docs, feature_generator))
        logger.info("GENERATING VALIDATION EXAMPLES")

        # logger.info("Checking polysemy of trainset:")
        # check_polysemy(train_examples)

        (dev_examples, dev_data, dev_data_list, dev_label) = (
            generate_trigger_data_feature(
                example_generator, test_docs, feature_generator))
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = generate_trigger_data_feature_from_serialized(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = generate_trigger_data_feature_from_serialized(example_generator, feature_generator, dev_candidates)
        (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature_from_serialized(example_generator, feature_generator, test_candidates)
        test_tuple = (test_examples, test_data, test_data_list, test_label)

    # Fit model
    trigger_model.fit_model(train_data_list, train_label, dev_data_list, dev_label)
    # TODO: review the following later

    if trigger_extractor.hyper_parameters.fine_tune_epoch > 0:
        if classifier_is_binary:
            raise NotImplementedError(
                "Fine-tuning epochs have not yet been implemented for binary "
                "models")
        for layer in trigger_model.model.layers:
            #if layer.name in (u'sentence_embedding', u'lex_embedding'):
            layer.trainable = True
        trigger_model.model.compile(optimizer=trigger_model.optimizer, loss=u'categorical_crossentropy', metrics=[])
        trigger_model.fit_model(train_data_list, train_label, dev_data_list, dev_label)

    # Save model data
    if params['save_model']:
        print('==== Saving Trigger model ====')
        trigger_model.save_keras_model(trigger_extractor.model_file)

    # Make validation-set predictions
    predictions = trigger_model.predict(dev_data_list)

    if classifier_is_binary:
        if trigger_extractor.use_trigger_safelist:
            # Modify predictions to account for non-positive trigger filtration
            apply_positive_training_trigger_filter_to_predictions(
                dev_examples, predictions, trigger_extractor)

        # Evaluate model using dev set
        logger.info("Evaluating multi-label validation-set performance ")
        best_thresholds = evaluate_baseline_and_best_multi_label_f1s(
            predictions, dev_label, trigger_extractor,
            params['train.score_file'])

        # Save thresholds, if possible
        if trigger_extractor.class_thresholds_path is not None:
            logger.info("Saving thresholds at {}"
                        .format(trigger_extractor.class_thresholds_path))
            np.savez(trigger_extractor.class_thresholds_path,
                     thresholds=best_thresholds)
            logger.info("Updating this extractor object's current thresholds")
            trigger_extractor.class_thresholds = best_thresholds

        # # Dump threshold analysis information for multi-label single-model
        # report_top_n_sample_info(50, predictions, test_examples, test_label,
        # best_thresholds, trigger_extractor, suffix='dev')

    else:
        # Get score
        logger.info('Single-label validation-set score:')
        score, score_breakdown = evaluate_f1(predictions, dev_label, trigger_extractor.domain.get_event_type_index('None'))
        logger.info(score.to_string())
        print_score_breakdown(trigger_extractor, score_breakdown)
        write_score_to_file(trigger_extractor, score, score_breakdown, params['train.score_file'])

    # get Predicted Positive Triggers for trigger info dump
    predicted_positive_triggers = get_predicted_positive_triggers(predictions, dev_examples, trigger_extractor)
    # { docid -> list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint] }

    # Report information about triggers
    single_word_prediction = 0
    multi_word_prediction = 0
    for docid in predicted_positive_triggers:
        for eg in predicted_positive_triggers[docid]:
            length = len(eg.anchor.tokens)
            if length == 1:
                single_word_prediction += 1
            elif length > 1:
                multi_word_prediction += 1
    logger.info('** #single_word_prediction={}, #multi_word_prediction={}'.format(single_word_prediction, multi_word_prediction))

    # logger.info("Checking polysemy of devset:")
    # check_polysemy(test_examples)

    # Also conduct testing if possible
    if 'test' in params['data']:
        if serialize_list is None:
            test_trigger(params, word_embeddings, trigger_extractor)
        else:
            test_trigger(params, word_embeddings, trigger_extractor, test_tuple=test_tuple)

def train_trigger_from_feature(params, extractor):
    train_data = np.load(params['data']['train']['features'])
    train_data_list = train_data['data_list']
    train_label = train_data['label']

    test_data = np.load(params['data']['dev']['features'])
    test_data_list = test_data['data_list']
    test_label = test_data['label']

    trigger_model = extractor.extraction_model
    """:type: nlplingo.model.trigger_model.TriggerModel"""

    trigger_model.fit_model(train_data_list, train_label, test_data_list, test_label)

    print('==== Saving Trigger model ====')
    trigger_model.save_keras_model(extractor.model_file)

# +1
def load_trigger_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'trigger.hdf'))
    return model
    #with open(os.path.join(model_dir, 'trigger.pickle'), u'rb') as f:
    #    trigger_model = pickle.load(f)
    #    """:type: nlplingo.model.event_cnn.ExtractionModel"""
        #trigger_model.load_keras_model(filename=os.path.join(model_dir, 'trigger.hdf'))
    #    trigger_model.model_dir = model_dir
    #    return trigger_model

# +1
def load_trigger_modelfile(filepath):
    return keras_load_model(str(filepath))


def test_trigger(params, word_embeddings, trigger_extractor, test_tuple=None):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    """

    feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator"""
    example_generator = trigger_extractor.example_generator
    """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""
    trigger_model = trigger_extractor.extraction_model
    """:type: nlplingo.nn.trigger_model.TriggerModel"""

    if test_tuple is None:
        test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

        for doc in test_docs:
            doc.apply_domain(trigger_extractor.domain)

        # Generate data
        logger.info("GENERATING TESTING EXAMPLES")
        (test_examples, test_data, test_data_list, test_label) = (
            generate_trigger_data_feature(
                example_generator, test_docs, feature_generator))
    else:
        test_examples, test_data, test_data_list, test_label = test_tuple

    # Make test-set predictions
    predictions = trigger_model.predict(test_data_list)

    if trigger_model.is_binary:
        if trigger_extractor.use_trigger_safelist:
            # Modify predictions to account for non-positive trigger filtration
            apply_positive_training_trigger_filter_to_predictions(
                test_examples, predictions, trigger_extractor)

        # Evaluate test set performance
        logger.info("Evaluating multi-label test-set performance")
        # Test-set oracle thresholds are not saved at this time.
        best_thresholds = evaluate_baseline_and_best_multi_label_f1s(
            predictions, test_label, trigger_extractor,
            params['test.score_file'])

        # # Dump threshold analysis information for multi-label single-model
        # report_top_n_sample_info(50, predictions, test_examples, test_label,
        # best_thresholds, trigger_extractor, suffix='test_oracle')

        # Tuned scoring (if available): tune-on-dev
        threshold_path = trigger_extractor.class_thresholds_path
        dev_thresholds = [0.5] * test_label.shape[1]
        if threshold_path is not None and os.path.isfile(threshold_path):
            # Get thresholds
            logger.info('Evaluating multi-label test-set performance using '
                        'tune-on-dev thresholds')
            dev_thresholds = np.load(threshold_path)['thresholds']

            # Evaluate
            score, score_breakdown = evaluate_multi_label_f1(
                predictions, test_label,
                trigger_extractor.domain.get_event_type_index('None'),
                thresholds=dev_thresholds)
            logger.info(score.to_string())
            print_score_breakdown(trigger_extractor, score_breakdown)
            write_score_to_file(
                trigger_extractor, score, score_breakdown,
                params['test.score_file'] + '.tune_on_dev')

            # # Dump threshold analysis information for multi-label single-model
            # report_top_n_sample_info(50, predictions, test_examples,
            # test_label, dev_thresholds, trigger_extractor, suffix='test')

        # Report detailed error information (before modifying the examples list)
        # - Prepare for report:
        label_info_per_sample = []
        for sample_idx in range(test_label.shape[0]):
            g_labels = set([
                trigger_extractor.domain.get_event_type_from_index(l_idx)
                for l_idx, label in enumerate(test_label[sample_idx])
                if label == 1])
            p_labels = set()
            for class_idx in range(test_label.shape[1]):
                score = predictions[sample_idx, class_idx]
                p_label = trigger_extractor.domain.get_event_type_from_index(
                    class_idx)
                if score >= dev_thresholds[class_idx]:
                    p_labels.add(p_label)
            label_info_per_sample.append((g_labels, p_labels))

        # - Recall errors
        for i, (g_labels, p_labels) in enumerate(label_info_per_sample):
            for g_label in sorted(g_labels):
                if g_label != 'None' and g_label not in p_labels:
                    # get anchor and sentence strings
                    a = test_examples[i].anchor.text.encode('ascii', 'ignore')
                    s = test_examples[i].sentence.text.encode('ascii', 'ignore')
                    logging.debug('RECALL-ERROR: {} {} {} {}'.format(
                        a, s, g_label, sorted(p_labels)))

        # - Precision errors
        for i, (g_labels, p_labels) in enumerate(label_info_per_sample):
            for p_label in sorted(p_labels):
                if p_label != 'None' and p_label not in g_labels:
                    # get anchor and sentence strings
                    a = test_examples[i].anchor.text.encode('ascii', 'ignore')
                    s = test_examples[i].sentence.text.encode('ascii', 'ignore')
                    logging.debug('PRECISION-ERROR: {} {} {} {}'.format(
                        a, s, sorted(g_labels), p_label))

        # Get predicted positive triggers
        predicted_positive_triggers = get_predicted_positive_triggers(
            predictions, test_examples, trigger_extractor)
        # @CLAY : should the above be filtered by dev_thresholds ?

    else:

        # Evaluate
        logger.info('Single-label test-set score:')
        score, score_breakdown = evaluate_f1(
            predictions, test_label,
            trigger_extractor.domain.get_event_type_index('None'))
        logger.info(score.to_string())
        print_score_breakdown(trigger_extractor, score_breakdown)
        write_score_to_file(trigger_extractor, score, score_breakdown, params['test.score_file'])

        # Get predicted positive triggers for error reporting purposes
        predicted_positive_triggers = get_predicted_positive_triggers(
            predictions, test_examples, trigger_extractor)

        # Report detailed error information
        # - Prepare for report:
        label_arg_max = np.argmax(test_label, axis=1)
        pred_arg_max = np.argmax(predictions, axis=1)

        # - Correct
        for i, v in enumerate(label_arg_max):
            g_label = trigger_extractor.domain.get_event_type_from_index(label_arg_max[i])
            p_label = trigger_extractor.domain.get_event_type_from_index(pred_arg_max[i])
            if g_label != 'None' and p_label == g_label:
                logging.debug('CORRECT-PREDICTIONS: {} {} {} gold={} pred={} {}'.format(
                    test_examples[i].sentence.docid,
                    test_examples[i].anchor.text.encode('ascii', 'ignore'),
                    test_examples[i].sentence.text.encode('ascii', 'ignore'),
                    g_label, p_label, '%.3f' % predictions[i][pred_arg_max[i]]))
        # - Recall errors
        for i, v in enumerate(label_arg_max):
            g_label = trigger_extractor.domain.get_event_type_from_index(label_arg_max[i])
            p_label = trigger_extractor.domain.get_event_type_from_index(pred_arg_max[i])
            if g_label != 'None' and p_label != g_label:
                logging.debug('RECALL-ERROR: {} {} {} gold={} pred={} {}'.format(
                    test_examples[i].sentence.docid,
                    test_examples[i].anchor.text.encode('ascii', 'ignore'),
                    test_examples[i].sentence.text.encode('ascii', 'ignore'),
                    g_label, p_label, '%.3f' % predictions[i][pred_arg_max[i]]))

        # - Precision errors
        for i, v in enumerate(label_arg_max):
            g_label = trigger_extractor.domain.get_event_type_from_index(label_arg_max[i])
            p_label = trigger_extractor.domain.get_event_type_from_index(pred_arg_max[i])
            if p_label != 'None' and p_label != g_label:
                logging.debug('PRECISION-ERROR: {} {} {} gold={} pred={} {}'.format(
                    test_examples[i].sentence.docid,
                    test_examples[i].anchor.text.encode('ascii', 'ignore'),
                    test_examples[i].sentence.text.encode('ascii', 'ignore'),
                    g_label, p_label, '%.3f' % predictions[i][pred_arg_max[i]]))

        confusion_counts, label_indices = calculate_confusion_matrix(pred_arg_max, label_arg_max)
        print_confusion_matrix_for_event_types(confusion_counts, label_indices, trigger_extractor.domain)

    # logger.info("Checking polysemy of testset:")
    # check_polysemy(test_examples)

    return predicted_positive_triggers


# def test_trigger_list(params, word_embeddings, event_domain):
#     """
#     :type params: nlplingo.common.parameters.Parameters
#     :type word_embeddings: nlplingo.embeddings.WordEmbedding
#     :type event_domain: nlplingo.tasks.event_domain.EventDomain
#     """
#
#     classifier_is_binary = trigger_model_is_binary(trigger_extractor.model_type)
#     if classifier_is_binary:
#         logger.debug("Warning: function test_trigger_list() does not have an "
#                      "implementation for a binary classifier!")
#
#     generator = EventTriggerExampleGenerator(event_domain, params)
#     event_keyword_list = EventKeywordList(params.get_string('trigger.event_keywords'))
#     event_keyword_list.print_statistics()
#
#     test_docs = prepare_docs(params.get_string('filelist.test'), word_embeddings)
#
#     for doc in test_docs:
#         doc.apply_domain(event_domain)
#     # constraint_event_type_to_domain(test_docs, event_domain)
#
#     examples = generator.generate(test_docs)
#     test_label = np.asarray([eg.label for eg in examples])
#
#     predictions = []
#     for eg in examples:
#         token = eg.anchor.tokens[0]
#
#         event_types = event_keyword_list.get_event_types_for_tokens([token])
#
#         event_type = 'None'
#         if len(event_types) == 1:
#             event_type = list(event_types)[0]
#         event_index = event_domain.get_event_type_index(event_type)
#
#         eg_predictions = np.zeros(len(event_domain.event_types), dtype=params.get_string('cnn.int_type'))
#         eg_predictions[event_index] = 1
#         predictions.append(eg_predictions)
#
#     number_of_recall_miss = 0
#     recall_misses = get_trigger_recall_misses(predictions, test_label, event_domain.get_event_type_index('None'),
#                                               event_domain, examples)
#     for key in sorted(recall_misses.keys()):
#         count = recall_misses[key]
#         if count > 0:
#             print('Trigger-recall-miss\t{}\t{}'.format(key, count))
#             number_of_recall_miss += count
#     print('Total# of recall miss={}'.format(number_of_recall_miss))
#
#     number_of_precision_miss = 0
#     precision_misses = get_trigger_precision_misses(predictions, test_label, event_domain.get_event_type_index('None'),
#                                                     event_domain, examples)
#     for key in sorted(precision_misses.keys()):
#         count = precision_misses[key]
#         if count > 0:
#             print('Trigger-precision-miss\t{}\t{}'.format(key, count))
#             number_of_precision_miss += count
#     print('Total# of precision miss={}'.format(number_of_precision_miss))
#
#     score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'))
#     print(score.to_string())
#
#     for index, f1_score in score_breakdown.items():
#         et = event_domain.get_event_type_from_index(index)
#         print('{}\t{}'.format(et, f1_score.to_string()))
#
#     output_dir = params.get_string('output_dir')
#     with open(os.path.join(output_dir, 'test_trigger.score'), 'w') as f:
#         f.write(score.to_string() + '\n')
#         for index, f1_score in score_breakdown.items():
#             et = event_domain.get_event_type_from_index(index)
#             f.write('{}\t{}\n'.format(et, f1_score.to_string()))


# def decode_trigger(params, word_embeddings, extractor):
#     """
#     :type params: dict
#     :type word_embeddings: nlplingo.embeddings.WordEmbedding
#     :type extractor: nlplingo.model.Extractor
#     """
#
#     classifier_is_binary = trigger_model_is_binary(trigger_extractor.model_type)
#     if classifier_is_binary:
#         logger.debug("Warning: function decode_trigger() does not have an "
#                      "implementation for a binary classifier!")
#         raise NotImplementedError
#
#     trigger_generator = extractor.generator
#
#     test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)
#
#     (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
#         trigger_generator,
#         test_docs,
#         extractor.model_type,
#         extractor.model_flags
#     )
#
#     predictions_output_file = params['predictions_file']
#     clusters = {}
#
#     print('==== Loading Trigger model ====')
#     trigger_model = load_trigger_modelfile(extractor.model_file)
#     trigger_predictions = trigger_model.predict(trigger_data_list)
#
#     predicted_positive_triggers = get_predicted_positive_triggers(
#         trigger_predictions,
#         trigger_examples,
#         extractor
#     )
#
#     for docid in predicted_positive_triggers:
#         for t in predicted_positive_triggers[docid]:
#             """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
#             print('PREDICTED-ANCHOR {} {} {} {}'.format(t.sentence.docid, t.event_type, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
#             cluster = clusters.setdefault(t.event_type, dict())
#             sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))), dict())
#             sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
#             sentence['eventType'] = t.event_type
#             sentence['docId'] = t.sentence.docid
#             sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
#             trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
#             trigger_array = trigger.setdefault('trigger', list())
#             trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
#             trigger_array = sorted(list(set(trigger_array)))
#             trigger['trigger'] = trigger_array
#
#     with open(predictions_output_file, 'w') as fp:
#             json.dump(clusters, fp, indent=4, sort_keys=True)


# def tune_trigger_thresholds_from_file(params, embedding_obj, extractor):
#     """
#     :type params: dict
#     :type embedding_obj: nlplingo.embeddings.word_embeddings.WordEmbedding
#     :type extractor: nlplingo.model.extractor.Extractor
#     """
#
#     if not os.path.isdir(os.path.dirname(extractor.model_file)):
#         raise IOError("No such directory {}"
#                       .format(os.path.dirname(extractor.model_file)))
#
#     classifier_is_binary = trigger_model_is_binary(extractor.model_type)
#     if classifier_is_binary:
#         logger.info("Model type has been identified as a binary classifier.")
#     else:
#         logger.info("Model type has been identified as a 1-vs-all classifier.")
#         logger.debug("This model type cannot have its thresholds tuned!")
#         raise IOError()
#
#     # grab components needed to use model
#     feature_generator = extractor.feature_generator
#     """:type: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator"""
#     example_generator = extractor.example_generator
#     """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""
#     trigger_model = extractor.extraction_model
#     """:type: nlplingo.nn.trigger_model.TriggerModel"""
#     logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
#     logger.debug('type(example_generator)={}'.format(type(example_generator)))
#     logger.debug('type(trigger_model)={}'.format(type(trigger_model)))
#
#     # prepare data
#     logger.info('==== Preparing dev docs ====')
#     dev_docs = prepare_docs(params['data']['dev']['filelist'], embedding_obj)
#     for doc in dev_docs:
#         doc.apply_domain(extractor.domain)
#     logger.info("GENERATING VALIDATION EXAMPLES")
#     (dev_examples, dev_data, dev_data_list, dev_label) = (
#         generate_trigger_data_feature(
#             example_generator, dev_docs, feature_generator))
#
#     # make predictions
#     predictions = trigger_model.predict(dev_data_list)
#     # Modify predictions to account for non-positive trigger filtration
#     apply_positive_training_trigger_filter_to_predictions(
#         dev_examples, predictions, extractor)
#
#     # Score baseline
#     logger.info('Multi-label validation-set score using untuned thresholds')
#     score, score_breakdown = evaluate_multi_label_f1(
#         predictions, dev_label,
#         extractor.domain.get_event_type_index('None'),
#         thresholds=None)
#     logger.info(score.to_string())
#     print_score_breakdown(extractor, score_breakdown)
#     write_score_to_file(extractor, score, score_breakdown,
#                         params['train.score_file'])
#
#     # Get tune-on-dev thresholds and save
#     logger.info('Tuning thresholds on dev set')
#     best_thresholds = calculate_all_thresholds(
#         predictions, dev_label, extractor.domain)
#     if extractor.class_thresholds_path is not None:
#         np.savez(extractor.class_thresholds_path, thresholds=best_thresholds)
#         extractor.class_thresholds = best_thresholds
#
#     # Evaluate again with tuned thresholds
#     logger.info('Multi-label validation-set score using tune-on-dev thresholds')
#     score, score_breakdown = evaluate_multi_label_f1(
#         predictions, dev_label,
#         extractor.domain.get_event_type_index('None'),
#         thresholds=best_thresholds)
#     logger.info(score.to_string())
#     print_score_breakdown(extractor, score_breakdown)
#     write_score_to_file(extractor, score, score_breakdown,
#                         params['train.score_file'] + '.tune_on_dev')

def active_learning_experiment(params, word_embeddings, extractor):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type extractor: nlplingo.model.extractor.Extractor
    """
    import math
    import random
    print('==== Preparing training docs ====')
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    print('==== Preparing dev docs ====')
    dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
    print('==== Preparing test docs ====')
    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    for doc in train_docs:
        doc.apply_domain(extractor.domain)
    for doc in dev_docs:
        doc.apply_domain(extractor.domain)
    for doc in test_docs:
        doc.apply_domain(extractor.domain)

    feature_generator = extractor.feature_generator
    """:type: nlplingo.tasks.eventtrigger.EventTriggerFeatureGenerator"""
    example_generator = extractor.example_generator
    """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""
    trigger_model = extractor.extraction_model
    """:type: nlplingo.model.trigger_model.TriggerModel"""

    (train_examples, train_data, train_data_list, train_label) = generate_trigger_data_feature(
        example_generator,
        train_docs,
        feature_generator
    )

    (dev_examples, dev_data, dev_data_list, dev_label) = generate_trigger_data_feature(
        example_generator,
        dev_docs,
        feature_generator
    )

    (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(
        example_generator,
        dev_docs,
        feature_generator
    )

    # Start here
    initial_size = int(math.ceil(0.10 * len(train_label)))
    budget = initial_size
    subset = random.sample(range(len(train_label)), initial_size)
    train_data_list_subset = [x[subset, :, :] for x in train_data_list]
    train_label_subset = train_label[subset, :]
    result_list = []

    activeBatchSelector = ActiveBatchSelectorFactory.createActiveBatchSelector(
        params['active_learning']['active_selector'], params['active_learning']
    )
    for i in range(9):
        trigger_model.fit_model(train_data_list_subset, train_label_subset, dev_data_list, dev_label)

        # scoring
        predictions = trigger_model.predict(test_data_list)
        score, score_breakdown = evaluate_f1(predictions, test_label, extractor.domain.get_event_type_index('None'))
        result_list.append((train_label_subset.shape[0], score.f1))

        #
        new_subset = activeBatchSelector.select(
            trigger_model,
            subset,
            train_label,
            train_data_list,
            budget
        )

        subset.extend(new_subset)
        train_data_list_subset = [x[subset, :, :] for x in train_data_list]
        train_label_subset = train_label[subset, :]

    results_file = params['active_learning']['results_file']

    with open(results_file, 'w') as f:
        for size, entry in result_list:
            f.write('{} {}\n'.format(size, entry))
        f.close()


def train_transformer_sequence_trigger(params, trigger_extractor):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    """
    bp_filepath = params['bp_file']
    keep_unannotated_sentences = trigger_extractor.extractor_params['keep_unannotated_sentences']
    tokenization_type = trigger_extractor.extractor_params['tokenization_type']     # SPACE, SERIF

    labels = []
    labels.extend('B-{}'.format(label) for label in sorted(trigger_extractor.domain.event_types.keys()) if label != 'None')
    labels.extend('I-{}'.format(label) for label in sorted(trigger_extractor.domain.event_types.keys()) if label != 'None')
    labels.append('O')

    bio_generator = BIOGenerator()

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: train_docs: list[nlplingo.text.text_theory.Document]"""

    bio_lines = bio_generator.generate_trigger_bio(bp_filepath, keep_unannotated_sentences, tokenization_type, train_docs)

    train_transformer_sequence(params, trigger_extractor.extractor_params, bio_lines, labels)
