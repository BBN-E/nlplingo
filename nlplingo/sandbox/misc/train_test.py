from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.common.scoring import evaluate_f1
from nlplingo.common.scoring import evaluate_f1_binary
from nlplingo.embeddings.word_embeddings import WordEmbeddingFactory
from nlplingo.tasks.eventargument.run import generate_argument_data_feature
from nlplingo.tasks.eventargument.run import test_argument
from nlplingo.tasks.eventargument.run import train_argument
from nlplingo.tasks.event_domain import EventDomain
from nlplingo.tasks.event_sentence import EventSentenceGenerator
from nlplingo.tasks.novel_event_type import NovelEventType
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator
from nlplingo.tasks.eventtrigger.run import generate_trigger_data_feature
from nlplingo.tasks.eventtrigger.run import get_predicted_positive_triggers
from nlplingo.tasks.eventtrigger.run import test_trigger
from nlplingo.tasks.eventtrigger.run import train_trigger_from_file
from nlplingo.model.sentence_model import MaxPoolEmbeddedSentenceModel
from nlplingo.nn.extractor import Extractor
from nlplingo.sandbox.misc.event_pair import EventPairData
from nlplingo.sandbox.misc.event_pair import EventPairGenerator
from nlplingo.sandbox.misc.event_pair import print_pair_predictions
from nlplingo.sandbox.model.pair_model import MaxPoolEmbeddedPairModel

logger = logging.getLogger(__name__)




def print_event_statistics(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    stats = defaultdict(int)
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                et = event.label
                stats[et] += 1
                for arg in event.arguments:
                    if arg.label == 'CyberAttackType' and et == 'CyberAttack':
                        print('CyberAttackType {} [{}] {}'.format(event.id, arg.text, sent.text.encode('ascii', 'ignore')))
                for anchor in event.anchors:
                    stats['{}.{}'.format(et, 'anchor')] += 1
                    if et == 'CyberAttack':
                        print('CyberAttack <{}> {}'.format(anchor.text, sent.text.encode('ascii', 'ignore')))
                for arg in event.arguments:
                    role = '{}.{}'.format(et, arg.label)
                    stats[role] += 1
    for key in sorted(stats.keys()):
        print('{}\t{}'.format(key, str(stats[key])))

# def constraint_event_type_to_domain(docs, event_domain):
#     """
#     :type docs: list[nlplingo.text.text_theory.Document]
#     :type event_domain: nlplingo.tasks.event_domain.EventDomain
#     """
#     for doc in docs:
#         doc.events = [tasks for tasks in doc.events if tasks.label in event_domain.event_types.keys()]
#     for doc in docs:
#         for sent in doc.sentences:
#             sent.events = [tasks for tasks in sent.events if tasks.label in event_domain.event_types.keys()]




def decode_trigger_argument(params, word_embeddings, trigger_extractors, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractors: list[nlplingo.nn.extractor.Extractor] # eventtrigger extractors
    :type argument_extractors: list[nlplingo.nnl.extractor.Extractor] # eventargument extractors
    """
    # Find the eventtrigger extractor

    trigger_extractor = None
    if len(trigger_extractors) > 1:
        raise RuntimeError('More than one eventtrigger model cannot be used in decoding.')
    elif len(trigger_extractors) == 1:
        trigger_extractor = trigger_extractors[0]

    if len(argument_extractors) == 0:
        raise RuntimeError('At least one eventargument extractor must be specified to decode over arguments.')

    if trigger_extractor is None:
        raise RuntimeError('Trigger extractor must be specified in parameter file.')

    trigger_generator = trigger_extractor.generator

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    logging.info('#### Generating eventtrigger examples')
    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_extractor.example_generator, test_docs, trigger_extractor.feature_generator)

    predicted_positive_triggers = []
    if len(trigger_examples) > 0:
        logging.info('#### Predicting triggers')
        trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)
        predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples,
                                                                  trigger_extractor.domain)

    predictions_output_file = params['predictions_file']
    clusters = {}

    for docid in predicted_positive_triggers:
        for t in predicted_positive_triggers[docid]:
            """:type: nlplingo.tasks.eventtrigger.example.EventTriggerExample"""
            logging.info('PREDICTED-ANCHOR {} {} {} {} {}'.format(t.sentence.docid, t.event_type, '%.4f' % t.score, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
            cluster = clusters.setdefault(t.event_type, dict())
            sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))), dict())
            sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
            sentence['eventType'] = t.event_type
            sentence['score'] = '%.4f' % (t.score)
            sentence['docId'] = t.sentence.docid
            sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
            trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
            trigger_array = trigger.setdefault('eventtrigger', list())
            trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
            trigger_array = sorted(list(set(trigger_array)))
            trigger['eventtrigger'] = trigger_array


    actor_ner_types = set(['PER', 'ORG', 'GPE'])
    place_ner_types = set(['GPE', 'FAC', 'LOC', 'ORG'])
    time_ner_types = set(['TIMEX2.TIME'])

    if len(predicted_positive_triggers) > 0:
        for argument_extractor in argument_extractors:
            logging.info('Generating eventargument examples based on predicted triggers')
            (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                generate_argument_data_feature(
                    argument_extractor.example_generator, test_docs, argument_extractor.feature_generator,
                    predicted_triggers=predicted_positive_triggers)

            if len(arg_examples_pt) == 0:
                continue

            logging.info('Predicting arguments based on predicted triggers')
            argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)
            pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

            for i, predicted_label in enumerate(pred_arg_max):
                if predicted_label != extractor.domain.get_event_role_index('None'):
                    eg = arg_examples_pt[i]
                    """:type: nlplingo.tasks.eventargument.example.EventArgumentExample"""
                    eg.score = argument_predictions_pt[i][predicted_label]
                    predicted_role = extractor.domain.get_event_role_from_index(predicted_label)

                    if predicted_role == 'Time' and eg.argument.label not in time_ner_types:
                        continue
                    if predicted_role == 'Place' and eg.argument.label not in place_ner_types:
                        continue
                    if predicted_role == 'Actor' and eg.argument.label not in actor_ner_types:
                        continue

                    print('PREDICTED-ARGUMENT {} {} {} {} {}'.format(eg.sentence.docid, predicted_role, '%.4f' % (eg.score), eg.argument.start_char_offset(), eg.argument.end_char_offset()))
                    cluster = clusters.setdefault(eg.anchor.label, dict())
                    sentence = cluster.setdefault(str((str(eg.sentence.docid), str(eg.sentence.int_pair.to_string()))), dict())
                    if sentence.get('token', None) is None:
                        print("Something is wrong")
                        sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(eg.sentence.tokens)]
                    trigger = sentence.setdefault('trigger_{}'.format(eg.anchor.int_pair.to_string()), dict())
                    argument = trigger.setdefault(predicted_role, list())
                    # eventargument.extend([tokenIdx.index_in_sentence for tokenIdx in eg.eventargument.tokens])
                    argument_array = [tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]
                    argument.append((min(argument_array), max(argument_array)))
                    argument = sorted(list(set(argument)))
                    trigger[predicted_role] = argument

    with open(predictions_output_file, 'w') as fp:
            json.dump(clusters, fp, indent=4, sort_keys=True)





# =============== Event sentence =================
def get_predicted_positive_sentences(predictions, examples, none_class_index, event_domain):
    """Collect the predicted positive sentences and organize them by docid
    Also, use the predicted event_type for each such sentence example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.tasks.event_sentence.EventSentenceExample]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    assert len(predictions)==len(examples)
    ret = defaultdict(list)

    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            eg = examples[i]
            """:type: nlplingo.tasks.event_sentence.EventSentenceExample"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            ret[eg.sentence.docid].append(eg)
    return ret

def generate_sentence_data_feature(generator, docs):
    """
    :type generator: nlplingo.tasks.event_sentence.EventSentenceGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    examples = generator.generate(docs)
    data = generator.examples_to_data_dict(examples)
    data_list = [np.asarray(data['word_vec'])]
    label = np.asarray(data['label'])

    print('#sentence-examples=%d' % (len(examples)))
    print('data word_vec.len=%d label.len=%d' % (len(data['word_vec']), len(data['label'])))
    return (examples, data, data_list, label)

def sentence_modeling(params, train_docs, test_docs, event_domain, word_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    """
    generator = EventSentenceGenerator(event_domain, params)

    (train_examples, train_data, train_data_list, train_label) = generate_sentence_data_feature(generator, train_docs)

    # train_examples = generator.generate(train_docs)
    # train_data = generator.examples_to_data_dict(train_examples)
    # train_data_list = [np.asarray(train_data['word_vec']), np.asarray(train_data['pos_array'])]
    # train_label = np.asarray(train_data['label'])
    #
    # print('#train_examples=%d' % (len(train_examples)))
    # print('train_data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    #     len(train_data['word_vec']), len(train_data['pos_array']), len(train_data['label'])))

    (test_examples, test_data, test_data_list, test_label) = generate_sentence_data_feature(generator, test_docs)

    # test_examples = generator.generate(test_docs)
    # test_data = generator.examples_to_data_dict(test_examples)
    # test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array'])]
    # test_label = np.asarray(test_data['label'])
    #
    # print('#test_examples=%d' % (len(test_examples)))
    # print('test_data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    #     len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['label'])))

    sentence_model = MaxPoolEmbeddedSentenceModel(params, event_domain, word_embeddings)
    sentence_model.fit(train_data_list, train_label, test_data_list, test_label)

    predictions = sentence_model.predict(test_data_list)
    predicted_positive_sentences = get_predicted_positive_sentences(predictions, test_examples, event_domain.get_event_type_index('None'), event_domain)

    # calculate the recall denominator
    number_test_positives = 0
    for doc in test_docs:
        for sentence in doc.sentences:
            labels = set()
            for event in sentence.events:
                labels.add(event.label)
            number_test_positives += len(labels)

    score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'), num_true=number_test_positives)
    print(score.to_string())

    output_dir = params.get_string('output_dir')
    with open(os.path.join(output_dir, 'train_sentence.score'), 'w') as f:
        f.write(score.to_string() + '\n')

    print('==== Saving Sentence model ====')
    sentence_model.save_keras_model(os.path.join(output_dir, 'sentence.hdf'))
    with open(os.path.join(output_dir, 'sentence.pickle'), u'wb') as f:
        pickle.dump(sentence_model, f)

    return predicted_positive_sentences


def train_sentence(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    train_docs = prepare_docs(params.get_string('filelist.train'))
    test_docs = prepare_docs(params.get_string('filelist.dev'))

    predicted_positive_sentences = sentence_modeling(params, train_docs, test_docs, event_domain, word_embeddings)


def to_example_pairs(examples, data, data_list, label):
    """
    :type examples: list[nlplingo.tasks.event_sentence.EventSentenceExample]
    :type data: defaultdict(list)
    :type data_list: [np.asarray(list)]
    :type label: np.asarray(list)
    """
    eg_by_label = defaultdict(list)
    for eg in examples:
        eg_by_label[eg.event_type].append(eg)

    for label in eg_by_label.keys():
        n = len(eg_by_label[label])
        n_choose_2 = (n * (n-1))/2
        print('{} combinations for label {}'.format(n_choose_2, label))

def generate_pair_data_from_triggers_pairs(egs1, egs2, pair_generator):
    """
    :type egs1: list[nlplingo.tasks.event_trigger.EventTriggerExample]
    :type egs2: list[nlplingo.tasks.event_trigger.EventTriggerExample]
    :type pair_generator: nlplingo.tasks.eventpair.EventPairGenerator
    """

    pair_examples = pair_generator.generate_cross_product(egs1, egs2)

    data = pair_generator.examples_to_data_dict(pair_examples)
    data_list = [np.asarray(data['word_vec1']), np.asarray(data['word_vec2']),
                 np.asarray(data['pos_data1']), np.asarray(data['pos_data2'])]
                 #np.asarray(data['word_cvec1']), np.asarray(data['word_cvec2']),
                 #np.asarray(data['dep_vec1']), np.asarray(data['dep_vec2'])]
    label = np.asarray(data['label'])
    #label = k_utils.to_categorical(np.array(data['label']), num_classes=2)

    print('data word_vec1.len=%d word_vec2.len=%d label.len=%d' % (len(data['word_vec1']), len(data['word_vec2']), len(data['label'])))
    return EventPairData(None, pair_examples, data, data_list, label)
    #return (pair_examples, data, data_list, label)

def generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, docs, training=False):
    """
    We set include_none=False during training, include_none=True when it is test data

    :type novel_event_type: nlplingo.tasks.novel_event_type.NovelEventType
    :type trigger_generator: nlplingo.tasks.event_trigger.EventTriggerGenerator
    :type pair_generator: nlplingo.tasks.eventpair.EventPairGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    """

    trigger_examples = trigger_generator.generate(docs)
    """:type: list[nlplingo.tasks.event_trigger.EventTriggerExample]"""

    egs = None
    """:type: list[nlplingo.tasks.event_trigger.EventTriggerExample]"""
    if training:
        egs = novel_event_type.filter_train(trigger_examples)
        et_count = defaultdict(int)
        for eg in egs:
            et_count[eg.event_type] += 1
        for et in et_count:
            print('In train egs: {} {}'.format(et, et_count[et]))

        pair_examples = pair_generator.generate_train(egs)
    else:
        #egs = novel_event_type.filter_test(trigger_examples)
        egs = trigger_examples
        print('novel_event_type.filter_test #trigger_examples={} #egs={}'.format(len(trigger_examples), len(egs)))
        et_count = defaultdict(int)
        for eg in egs:
            et_count[eg.event_type] += 1
        for et in et_count:
            print('In test egs: {} {}'.format(et, et_count[et]))

        pair_examples = pair_generator.generate_test(egs)


    data = pair_generator.examples_to_data_dict(pair_examples)
    data_list = [np.asarray(data['word_vec1']), np.asarray(data['word_vec2']),
                 np.asarray(data['pos_data1']), np.asarray(data['pos_data2'])]
                 #np.asarray(data['word_cvec1']), np.asarray(data['word_cvec2']),
                 #np.asarray(data['dep_vec1']), np.asarray(data['dep_vec2'])]
    label = np.asarray(data['label'])
    #label = k_utils.to_categorical(np.array(data['label']), num_classes=2)

    print('data word_vec1.len=%d word_vec2.len=%d label.len=%d' % (len(data['word_vec1']), len(data['word_vec2']), len(data['label'])))

    return EventPairData(egs, pair_examples, data, data_list, label)
    #return (egs, pair_examples, data, data_list, label)


def calculate_pair_true_positive(docs, dataset_prefix, target_event_types):
    """We need this because in generating eventtrigger candidates, we heuristically reject some candidates
    So we cannot calculate the true positive from the candidates. We need to go back to the doc level annotations.

    :type docs: list[nlplingo.text.text_theory.Document]
    :type dataset_prefix: str
    :type target_event_types: set[str]
    """
    event_count = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            if event.label in target_event_types:
                event_count[event.label] += 1
    pair_true_positive = 0
    for et in event_count:
        print('In {} docs: {} {}'.format(dataset_prefix, et, event_count[et]))
        count = event_count[et]
        pair_true_positive += (count * (count - 1)) / 2
    return pair_true_positive

def pair_modeling(params, train_docs, dev_docs, test_docs, event_domain, word_embeddings, causal_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type dev_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type causal_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    """

    novel_event_type = NovelEventType(params)

    trigger_generator = EventTriggerExampleGenerator(event_domain, params)
    pair_generator = EventPairGenerator(event_domain, params, word_embeddings)


    print('#### Generating Training data')
    #(train_triggers, train_examples, train_data, train_data_list, train_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, train_docs, training=True)
    train_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, train_docs, training=True)
    # train_triggers: list[nlplingo.tasks.event_trigger.EventTriggerExample]

    train_triggers_new_types = [eg for eg in train_data.trigger_examples if eg.event_type in novel_event_type.new_types]
    """:type: list[nlplingo.tasks.event_trigger.EventTriggerExample]"""

    print('#### Generating Dev data')
    #(dev_triggers, dev_examples, dev_data, dev_data_list, dev_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, dev_docs, training=False)
    dev_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, dev_docs, training=False)

    print('#### Generating Test data')
    #(test_triggers, test_examples, test_data, test_data_list, test_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, test_docs, training=False)
    test_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, test_docs, training=False)

    # The idea is in training data, we have annotated very few eventtrigger examples of the new tasks types
    # We want to take the cross product between these few training examples, and the test eventtrigger candidates
    # Later on, we can then feed the pairwise probabilities into some heuristics to select test eventtrigger candidates that are most similar to the traininig examples
    train_test_data = generate_pair_data_from_triggers_pairs(train_triggers_new_types, test_data.trigger_examples, pair_generator)

    pair_model = MaxPoolEmbeddedPairModel(params, event_domain, word_embeddings, causal_embeddings)
    print('** train_data_list')
    print(train_data.data_list)
    print('** train_label')
    print(train_data.label)
    pair_model.fit(train_data.data_list, train_data.label, dev_data.data_list, dev_data.label)



    #print(predictions)

    #print('len(test_label)={}'.format(len(test_label)))
    #print('len(predictions)={}'.format(len(predictions)))

    #accuracy = evaluate_accuracy(predictions, test_label)
    #print('Accuracy=%.2f' % (accuracy))

    # class_labels = []
    # for eg in test_examples:
    #     if eg.label_string == 'SAME':
    #         class_labels.append(eg.eg1.event_type)
    #     else:
    #         class_labels.append('None')





    #### score on dev
    dev_predictions = pair_model.predict(dev_data.data_list)
    dev_tp_existing_type = calculate_pair_true_positive(dev_docs, 'dev', novel_event_type.existing_types)
    dev_tp_new_type = calculate_pair_true_positive(dev_docs, 'dev', novel_event_type.new_types)

    f1_dict_existing = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, novel_event_type.existing_types, dev_tp_existing_type)
    for f1 in f1_dict_existing:
        print('Dev Existing-type F1 score: {}'.format(f1.to_string()))
    f1_dict_new = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, novel_event_type.new_types, dev_tp_new_type)
    for f1 in f1_dict_new:
        print('Dev New-type F1 score: {}'.format(f1.to_string()))
    with open(params.get_string('output_dir') + '/dev.score', 'w') as f:
        for f1 in f1_dict_existing:
            f.write('Existing-type F1 score: {}\n'.format(f1.to_string()))
        for f1 in f1_dict_new:
            f.write('New-type F1 score: {}\n'.format(f1.to_string()))

    #### score on test
    test_predictions = pair_model.predict(test_data.data_list)
    test_tp_existing_type = calculate_pair_true_positive(test_docs, 'test', novel_event_type.existing_types)
    test_tp_new_type = calculate_pair_true_positive(test_docs, 'test', novel_event_type.new_types)

    f1_dict_existing = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, novel_event_type.existing_types, test_tp_existing_type)
    for f1 in f1_dict_existing:
        print('Test Existing-type F1 score: {}'.format(f1.to_string()))
    f1_dict_new = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, novel_event_type.new_types, test_tp_new_type)
    for f1 in f1_dict_new:
        print('Test New-type F1 score: {}'.format(f1.to_string()))
    f1_test_lines = []
    for et in novel_event_type.new_types:
        tp = calculate_pair_true_positive(test_docs, 'test', et)
        f1 = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, set([et]), tp, et)
        f1_test_lines.append(f1[0].to_string())
    with open(params.get_string('output_dir') + '/test.score', 'w') as f:
        for f1 in f1_dict_existing:
            f.write('Existing-type F1 score: {}\n'.format(f1.to_string()))
        for f1 in f1_dict_new:
            f.write('New-type F1 score: {}\n'.format(f1.to_string()))
        for score in f1_test_lines:
            f.write('F1 score: {}\n'.format(score))

    #### score on train-test
    train_test_predictions = pair_model.predict(train_test_data.data_list)
    f1_train_test_lines = []
    f1_train_test = evaluate_f1_binary(train_test_predictions, train_test_data.label, train_test_data.pair_examples, novel_event_type.new_types, None)
    f1_train_test_lines.append(f1_train_test[0].to_string())
    for et in novel_event_type.new_types:
        f1 = evaluate_f1_binary(train_test_predictions, train_test_data.label, train_test_data.pair_examples, set([et]), None, et)
        f1_train_test_lines.append(f1[0].to_string())
    with open(params.get_string('output_dir') + '/train_test.score', 'w') as f:
        for score in f1_train_test_lines:
            f.write('F1 score: {}\n'.format(score))
    print_pair_predictions(train_test_data.pair_examples, train_test_predictions, params.get_string('output_dir')+'/train_test.predictions')

                # output_dir = params.get_string('output_dir')
    # with open(os.path.join(output_dir, 'train_pair.score'), 'w') as f:
    #     f.write('Accuracy={}\n'.format(accuracy))
    #
    # print('==== Saving Pair model ====')
    # pair_model.save_keras_model(os.path.join(output_dir, 'pair.hdf'))
    # with open(os.path.join(output_dir, 'pair.pickle'), u'wb') as f:
    #     pickle.dump(pair_model, f)


def train_pair(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    train_docs = prepare_docs(params.get_string('filelist.train'), word_embeddings)
    dev_docs = prepare_docs(params.get_string('filelist.dev'), word_embeddings)
    test_docs = prepare_docs(params.get_string('filelist.test'), word_embeddings)

    pair_modeling(params, train_docs, dev_docs, test_docs, event_domain, word_embeddings)


def generate_event_statistics(params, word_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)

    anchor_types = defaultdict(int)
    anchor_lines = []
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                anchor = event.anchors[0]
                anchor_types[anchor.label] += 1
                anchor_lines.append('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(doc.docid, anchor.id, anchor.label, anchor.text, anchor.head().text, anchor.start_char_offset(), anchor.end_char_offset()))

    with codecs.open(params['output.event_type_count'], 'w', encoding='utf-8') as o:
        for et in sorted(anchor_types):
            o.write('{}\t{}\n'.format(et, anchor_types[et]))

    with codecs.open(params['output.anchor_info'], 'w', encoding='utf-8') as o:
        for l in anchor_lines:
            o.write(l)
            o.write('\n')


def load_embeddings(params):
    """
    :return: dict[str : WordEmbeddingAbstract]
    """
    embeddings = dict()

    if 'embeddings' in params:
        embeddings_params = params['embeddings']
        word_embeddings = WordEmbeddingFactory.createWordEmbedding(
            embeddings_params.get('type', 'word_embeddings'),
            embeddings_params
        )
        embeddings['word_embeddings'] = word_embeddings
        print('Word embeddings loaded')

    if 'dependency_embeddings' in params:
        dep_embeddings_params = params['dependency_embeddings']
        dependency_embeddings = WordEmbeddingFactory.createWordEmbedding(
            dep_embeddings_params.get('type', 'dependency_embeddings'),
            dep_embeddings_params
        )
        embeddings['dependency_embeddings'] = dependency_embeddings
        print('Dependency embeddings loaded')

    return embeddings


# unrunnable in current state
def decode_trigger_argument_for_active_learning(params, word_embeddings, extractors, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type extractors: list[nlplingo.model.extractor.Extractor] # trigger extractors
    :type argument_extractors: list[nlplingo.model.extractor.Extractor] # argument extractors
    """
    # Find the trigger extractor

    trigger_extractor = None
    if len(extractors) > 1:
        raise RuntimeError('More than one trigger model cannot be used in decoding.')
    elif len(extractors) == 1:
        trigger_extractor = extractors[0]

    if len(argument_extractors) == 0:
        raise RuntimeError('At least one argument extractor must be specified to decode over arguments. {}'.format(len(extractors)))

    if trigger_extractor is None:
        raise RuntimeError('Trigger extractor must be specified in parameter file.')

    trigger_generator = trigger_extractor.generator

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_generator,
        test_docs,
        trigger_extractor.model_type,
        trigger_extractor.model_flags
    )

    predictions_output_file = params['predictions_file']
    clusters = {}

    print('==== Loading Trigger model ====')
    trigger_model = trigger_extractor.extraction_model
    predicted_positive_triggers = []
    if len(trigger_examples) > 0:
        trigger_predictions = trigger_model.predict(trigger_data_list)
        predicted_positive_triggers = get_predicted_positive_triggers_with_active_learning_metric(
            trigger_predictions,
            trigger_examples,
            trigger_extractor.domain.get_event_type_index('None'),
            trigger_extractor.domain,
            best_vs_second_best
        )

    for docid in predicted_positive_triggers:
        for t in predicted_positive_triggers[docid]:
            """:type: nlplingo.tasks.eventtrigger.EventTriggerExample"""
            print('PREDICTED-ANCHOR {} {} {} {} {}'.format(t.sentence.docid, t.event_type, '%.4f' % t.score, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
            cluster = clusters.setdefault(t.event_type, dict())
            sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))), dict())
            sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
            sentence['eventType'] = t.event_type
            sentence['score'] = '%.4f' % (t.score)
            sentence['docId'] = t.sentence.docid
            sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
            trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
            trigger_array = trigger.setdefault('trigger', list())
            trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
            trigger_array = sorted(list(set(trigger_array)))
            trigger['trigger'] = trigger_array
            trigger['active_learning_raw_score'] = float(t.active_learning_raw_score)

    for extractor in argument_extractors:
        print('Loading argument model {}'.format(extractor.model_file))
        argument_model = extractor.extraction_model

        if len(predicted_positive_triggers) > 0:
            # generate arguments with predicted triggers
            (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                generate_argument_data_feature(
                    extractor.generator,
                    test_docs,
                    extractor.model_type,
                    extractor.model_flags,
                    predicted_triggers=predicted_positive_triggers
                )

            pred_arg_max = []
            if len(arg_examples_pt) > 0:
                # decode arguments with predicted triggers
                argument_predictions_pt = argument_model.predict(arg_data_list_pt)
                pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

            for i, predicted_label in enumerate(pred_arg_max):
                #if predicted_label != event_domain.get_event_role_index('None'):
                if predicted_label != extractor.domain.get_event_role_index('None'):
                    eg = arg_examples_pt[i]
                    """:type: nlplingo.tasks.eventargument.EventArgumentExample"""
                    eg.score = argument_predictions_pt[i][predicted_label]
                    #predicted_role = event_domain.get_event_role_from_index(predicted_label)
                    predicted_role = extractor.domain.get_event_role_from_index(predicted_label)

                    if predicted_role == 'Time' and eg.argument.label != 'TIMEX2.TIME':
                        continue
                    if predicted_role == 'Place' and not (eg.argument.label.startswith('GPE') or eg.argument.label.startswith(
                            'FAC') or eg.argument.label.startswith('LOC') or eg.argument.label.startswith('ORG')):
                        continue
                    if predicted_role == 'Actor' and not (eg.argument.label.startswith('PER') or eg.argument.label.startswith(
                            'ORG') or eg.argument.label.startswith('GPE')):
                        continue

                    print('PREDICTED-ARGUMENT {} {} {} {} {}'.format(eg.sentence.docid, predicted_role, '%.4f' % (eg.score), eg.argument.start_char_offset(), eg.argument.end_char_offset()))
                    cluster = clusters.setdefault(eg.anchor.label, dict())
                    sentence = cluster.setdefault(str((str(eg.sentence.docid), str(eg.sentence.int_pair.to_string()))), dict())
                    if sentence.get('token', None) is None:
                        print("Something is wrong")
                        sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(eg.sentence.tokens)]
                    trigger = sentence.setdefault('trigger_{}'.format(eg.anchor.int_pair.to_string()), dict())
                    argument = trigger.setdefault(predicted_role, list())
                    # argument.extend([tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens])
                    argument_array = [tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]
                    argument.append((min(argument_array), max(argument_array)))
                    argument = sorted(list(set(argument)))
                    # if eg.sentence.docid == 'ENG_NW_NODATE_0001':
                    #     print("predicted_role:{},current_anchor:{},current_word:{},array:{}".format(predicted_role,eg.anchor.text,eg.argument.text,[tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]))
                    trigger[predicted_role] = argument

    with open(predictions_output_file, 'w') as fp:
            json.dump(clusters, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)

    load_extractor_models_from_file = False
    if args.mode in {'test_trigger', 'test_argument', 'decode_trigger_argument', 'decode_trigger'}:
        load_extractor_models_from_file = True

    trigger_extractors = []
    argument_extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractor = Extractor(params, extractor_params, embeddings, load_extractor_models_from_file)
        if extractor.model_type.startswith('tasks-trigger_'):
            trigger_extractors.append(extractor)
        elif extractor.model_type.startswith('tasks-argument_'):
            argument_extractors.append(extractor)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(extractor.model_type))

    # event_domain = None
    # if params.get_string('domain') == 'general':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), params.get_string('domain'))
    # elif params.get_string('domain') == 'cyber':
    #     event_domain = CyberDomain()
    # elif params.get_string('domain') == 'ace':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ace')
    # elif params.get_string('domain') == 'precursor':
    #     event_domain = PrecursorDomain()
    # elif params.get_string('domain') == 'ui':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')
    # elif params.get_string('domain') == 'ace-precursor':
    #     event_domain = AcePrecursorDomain()
    # elif params.get_string('domain') == 'cyberattack':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')

    if 'domain_ontology.scoring' in params:
        scoring_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology.scoring'), 'scoring')
    else:
        scoring_domain = None


    #print(event_domain.to_string())

    if args.mode == 'train_trigger_from_file':
        train_trigger_from_file(params, embeddings, trigger_extractors[0])
    elif args.mode == 'train_trigger_from_model':
        train_trigger(params, word_embeddings, extractors[0], from_model=True)

    elif args.mode == 'train_trigger_from_feature':
        train_trigger_from_feature(params, extractors[0])
    elif args.mode == 'test_trigger':
        test_trigger(params, embeddings, trigger_extractors[0])

    elif args.mode == 'train_argument':
        train_argument(params, embeddings, argument_extractors[0])
    elif args.mode == 'test_trigger_list':
        test_trigger_list(params, word_embeddings, event_domain)
    elif args.mode == 'test_argument':
        test_argument(params, embeddings, trigger_extractors[0], argument_extractors[0], scoring_domain)
    elif args.mode == 'train_sentence':
        train_sentence(params, word_embeddings, event_domain)
    elif args.mode == 'train_pair':
        train_pair(params, word_embeddings, event_domain)
    elif args.mode == 'decode_trigger_argument':
        decode_trigger_argument(params, embeddings, trigger_extractors[0], argument_extractors)
    elif args.mode == 'decode_trigger':
        decode_trigger(params, word_embeddings, extractors[0])
    elif args.mode == 'event_statistics':
        generate_event_statistics(params, word_embeddings)
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))
