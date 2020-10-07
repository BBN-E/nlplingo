from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json
import logging
import random
import codecs
import copy

from collections import defaultdict

import numpy as np
from nlplingo.embeddings.word_embeddings import load_embeddings

from keras.models import load_model as keras_load_model

from nlplingo.common.scoring import evaluate_f1
from nlplingo.common.scoring import evaluate_multi_label_f1
from nlplingo.common.scoring import evaluate_baseline_and_best_multi_label_f1s
from nlplingo.common.scoring import print_score_breakdown
from nlplingo.common.scoring import write_score_to_file
from nlplingo.common.scoring import report_top_n_sample_info
from nlplingo.common.scoring import calculate_all_thresholds
from nlplingo.common.utils import F1Score

from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator


from nlplingo.tasks.eventtrigger.generator import EventKeywordList
from nlplingo.tasks.eventtrigger.metric import get_recall_misses as get_trigger_recall_misses
from nlplingo.tasks.eventtrigger.metric import get_precision_misses as get_trigger_precision_misses
from nlplingo.nn.extractor import Extractor

logger = logging.getLogger(__name__)

# +1
class EventPairData(object):
    def __init__(self, trigger_examples, pair_examples, data, data_list, label):
        """
        :type trigger_examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        :type pair_examples: list[nlplingo.tasks.eventpair.example.EventPairExample]
        :type data: defaultdict[str, list[numpy.ndarray]]
        :type data_list: list[numpy.ndarray]
        :type label: numpy.ndarray
        """
        self.trigger_examples = trigger_examples
        self.pair_examples = pair_examples
        self.data = data
        self.data_list = data_list
        self.label = label


class CorpusExampleInfo(object):
    """
    For each unlabeled/corpus event trigger, we create a CorpusExampleInfo object

    self.predictions is a dict[str]->list[float].
    This helps us calculate an average probability of being relevant to each (event-type) query set.
    For each event-type, we store the list of prediction-probabilities.
    For instance, if there is a total of 2 event-types, and 10 queries per event-type, then len(self.predictions.keys())==2,
    and each of those key/event-type will point to a list of 10 prediction probabilities
    This allows us to later use the function self.calculate_et_predictions() to average the list of probabilities,
    so that we get an averaged prediction-probability per event-type.

    And then we store the above averages (for each event-type) in self.et_predictions.

    self.labels and self.et_labels are to enable scoring.
    """
    def __init__(self, id):
        self.id = id
        self.predictions = defaultdict(list)
        self.labels = defaultdict(list)
        self.et_predictions = dict()    # averaged prediction values, across all queries with the same event type
        self.et_labels = dict()

    def calculate_et_predictions(self):
        for label in self.predictions:
            self.et_predictions[label] = np.average(self.predictions[label])

    def assert_labels_are_consistent(self):
        for label in self.labels:
            assert np.unique(self.labels[label]).size == 1

    def calculate_et_labels(self):
        for label in self.labels:
            self.et_labels[label] = np.unique(self.labels[label])[0]


def calculate_scores_by_query_event_type(pair_examples, predictions, labels, outfile_path, logfile):
    """
    :type pair_examples: list[nlplingo.tasks.eventpair.example.EventPairExample]
    """

    id_to_event_trigger = dict()    # we use this later to print out prediction rankings

    query_event_types = set()
    eg_infos = dict()
    """:type: dict[str, CorpusExampleInfo]"""
    for i, example in enumerate(pair_examples):
        query = example.eg1
        query_event_types.add(query.event_type)

        corpus_em = example.eg2
        corpus_em_id = '%s_%d-%d' % (corpus_em.sentence.docid, corpus_em.anchor.start_char_offset(), corpus_em.anchor.end_char_offset())
        id_to_event_trigger[corpus_em_id] = corpus_em

        if corpus_em_id not in eg_infos:
            eg_info = CorpusExampleInfo(corpus_em_id)
        else:
            eg_info = eg_infos[corpus_em_id]

        eg_info.predictions[query.event_type].append(predictions[i])
        eg_info.labels[query.event_type].append(labels[i])
        eg_infos[corpus_em_id] = eg_info

    # we now calculate the averaged prediction probabilities for each query (event-type) set
    for em_id in eg_infos:
        eg_info = eg_infos[em_id]
        eg_info.calculate_et_predictions()
        eg_info.assert_labels_are_consistent()
        eg_info.calculate_et_labels()

    log_lines = []
    map_scores = dict()
    for et in query_event_types:
        et_prediction_list = []
        et_label_list = []
        for eg in eg_infos.values():
            if et in eg.et_predictions and et in eg.et_labels:
                et_prediction_list.append(eg.et_predictions[et])
                et_label_list.append(eg.et_labels[et])

                event_trigger_example = id_to_event_trigger[eg.id]
                log_lines.append('query-type=%s similarity=%.3f gold-label=%s trigger-type=%s id=%s' % (et, eg.et_predictions[et], eg.et_labels[et], event_trigger_example.event_type, eg.id))

        map_score = mAP(et_label_list, et_prediction_list)
        map_scores[et] = map_score

    macro_map_score = np.average(list(map_scores.values()))

    outlines = []
    outlines.append('#### MAP scores')
    outlines.append('MAP_score: macro_average %.3f' % (macro_map_score))
    for et in sorted(map_scores):
        outlines.append('MAP_score: %s %.3f' % (et, map_scores[et]))

    with codecs.open(outfile_path, 'w', encoding='utf-8') as o:
        for line in outlines:
            o.write(line)
            o.write('\n')

    with codecs.open(logfile, 'w', encoding='utf-8') as o:
        for line in log_lines:
            o.write(line)
            o.write('\n')


def calculate_cosine_similarity(pair_examples):
    """
    :type pair_examples: list[nlplingo.tasks.eventpair.example.EventPairExample]
    """
    ret = []
    for pair_example in pair_examples:
        v1 = pair_example.eg1.anchor.head().word_vector
        v2 = pair_example.eg2.anchor.head().word_vector
        cos_sim = np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
        ret.append(cos_sim)
    return ret


def train_eventpair(params, word_embeddings, trigger_extractor, eventpair_extractor, pred_threshold):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type eventpair_extractor: nlplingo.nn.extractor.Extractor
    """

    # prepare dataset for sample generation
    logger.info("Preparing docs")
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
    logger.info("Applying domain")
    for doc in train_docs + dev_docs + test_docs:
        doc.apply_domain(eventpair_extractor.domain)

    print('#### Generating Training data')
    train_data = generate_pair_data_feature(trigger_extractor, eventpair_extractor, train_docs, 'training')
    """:type: nlplingo.tasks.eventpair.run.EventPairData"""

    print('#### Generating Dev data')
    dev_data = generate_pair_data_feature(trigger_extractor, eventpair_extractor, dev_docs, 'dev')
    """:type: nlplingo.tasks.eventpair.run.EventPairData"""

    print('#### Generating Test data')
    test_data = generate_pair_data_feature(trigger_extractor, eventpair_extractor, test_docs, 'test')
    """:type: nlplingo.tasks.eventpair.run.EventPairData"""

    # The idea is in training data, we have annotated very few trigger examples of the new event types
    # We want to take the cross product between these few training examples, and the test trigger candidates
    # Later on, we can then feed the pairwise probabilities into some heuristics to select test trigger candidates that are most similar to the traininig examples
    #train_test_data = generate_pair_data_from_triggers_pairs(train_triggers_new_types, test_data.trigger_examples, pair_generator)

    eventpair_model = eventpair_extractor.extraction_model
    """:type: nlplingo.nn.eventpair_model.EventPairModel"""
    #eventpair_model.fit_model(train_data.data_list, train_data.label, dev_data.data_list, dev_data.label)
    eventpair_model.fit_model(train_data.data_list, train_data.label, [], [])     # forgo validation during training epoch, to save compute time

    # Save model data
    if params['save_model']:
        print('==== Saving EventPair model ====')
        eventpair_model.save_keras_model(eventpair_extractor.model_file)

    # ==== dev data scoring ====
    dev_predictions = eventpair_model.predict(dev_data.data_list)
    dev_cos_sim = calculate_cosine_similarity(dev_data.pair_examples)
    print('Cosine Dev min=%.3f max=%.3f' % (np.min(dev_cos_sim), np.max(dev_cos_sim)))

    f1_dev = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, pred_threshold)
    map_dev = mAP(dev_data.label, dev_predictions)

    cos_f1_dev = evaluate_f1_binary(dev_cos_sim, dev_data.label, dev_data.pair_examples, pred_threshold=0.5)
    cos_map_dev = mAP(dev_data.label, dev_cos_sim)

    # we can only do the following if we are not sampling from the dev examples
    #dev_tp = calculate_pair_true_positive(dev_docs, 'dev')
    #f1_dev = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, dev_tp)

    for f1 in f1_dev:
        print('Dev F1 score: {}\tMAP: {}'.format(f1.to_string(), map_dev))
    with open(params['train.score_file'], 'w') as o:
        for f1 in f1_dev:
            o.write('F1 score: {}\tMAP: {}\n'.format(f1.to_string(), map_dev))

    for f1 in cos_f1_dev:
        print('Cosine Dev F1 score: {}\tMAP: {}'.format(f1.to_string(), cos_map_dev))
    with open(params['train.cosine_score_file'], 'w') as o:
        for f1 in cos_f1_dev:
            o.write('F1 score: {}\tMAP: {}\n'.format(f1.to_string(), cos_map_dev))

    # ==== test data scoring ====
    test_predictions = eventpair_model.predict(test_data.data_list)
    test_cos_sim = calculate_cosine_similarity(test_data.pair_examples)
    print('Cosine Test min=%.3f max=%.3f' % (np.min(test_cos_sim), np.max(test_cos_sim)))

    f1_test = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, pred_threshold)
    map_test = mAP(test_data.label, test_predictions)

    cos_f1_test = evaluate_f1_binary(test_cos_sim, test_data.label, test_data.pair_examples, pred_threshold=0.5)
    cos_map_test = mAP(test_data.label, test_cos_sim)

    # we can only do the following if we are not sampling from the test examples
    #test_tp = calculate_pair_true_positive(test_docs, 'test')
    #f1_test = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, test_tp)

    for f1 in f1_test:
        print('Test F1 score: {}\tMAP: {}'.format(f1.to_string(), map_test))
    with open(params['test.score_file'], 'w') as o:
        for f1 in f1_test:
            o.write('F1 score: {}\tMAP: {}\n'.format(f1.to_string(), map_test))

    for f1 in cos_f1_test:
        print('Cosine Test F1 score: {}\tMAP: {}'.format(f1.to_string(), cos_map_test))
    with open(params['test.cosine_score_file'], 'w') as o:
        for f1 in cos_f1_test:
            o.write('F1 score: {}\tMAP: {}\n'.format(f1.to_string(), cos_map_test))

    calculate_scores_by_query_event_type(test_data.pair_examples, test_predictions, test_data.label, params['test.collated.score_file'], params['test.collated.log'])
    calculate_scores_by_query_event_type(test_data.pair_examples, test_cos_sim, test_data.label, params['test.collated.cosine_score_file'], params['test.collated.cosine_log'])

    with open(params['information_file'], 'w', encoding='utf-8') as o:
        for line in eventpair_extractor.example_generator.information:
            o.write(line)
            o.write('\n')


def generate_pair_data_from_triggers_pairs(egs1, egs2, pair_generator):
    """
    :type egs1: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
    :type egs2: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
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


# +1
def generate_pair_data_feature(trigger_extractor, eventpair_extractor, docs, data_split):
    """
    We set include_none=False during training, include_none=True when it is test data
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type eventpair_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """

    trigger_feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator"""
    trigger_example_generator = trigger_extractor.example_generator
    """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""

    eventpair_feature_generator = eventpair_extractor.feature_generator
    """:type: nlplingo.tasks.eventpair.feature.EventPairFeatureGenerator"""
    eventpair_example_generator = eventpair_extractor.example_generator
    """:type: nlplingo.tasks.eventpair.generator.EventPairExampleGenerator"""

    # generate event trigger examples from the documents
    trigger_examples = trigger_example_generator.generate(docs, trigger_feature_generator)
    """:type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""

    if data_split == 'training':
        pair_examples = eventpair_example_generator.generate_train(trigger_examples, eventpair_example_generator.max_train_samples_per_class)
    elif data_split == 'dev':
        # we generate all pairs of trigger examples
        pair_examples = eventpair_example_generator.generate_dev(trigger_examples, eventpair_example_generator.max_dev_samples_per_class)
    elif data_split == 'test':
        pair_examples = eventpair_example_generator.generate_test(trigger_examples, eventpair_example_generator.max_test_samples_per_class)

    data = eventpair_example_generator.examples_to_data_dict(pair_examples, eventpair_feature_generator.features)

    data_list = [np.asarray(data[k]) for k in eventpair_feature_generator.features.feature_strings]
    label = np.asarray(data['label'])

    return EventPairData(trigger_examples, pair_examples, data, data_list, label)


# +1
def calculate_pair_true_positive(docs, dataset_prefix, target_event_types=None):
    """We need this because in generating trigger candidates, we heuristically reject some candidates
    So we cannot calculate the true positive from the candidates. We need to go back to the doc level annotations.

    :type docs: list[nlplingo.text.text_theory.Document]
    :type dataset_prefix: str
    :type target_event_types: set[str]  # This allows you to just test on some types, e.g. when you want to train on known types and test on novel types
    """
    event_count = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            if target_event_types is None or event.label in target_event_types:
                event_count[event.label] += 1
    pair_true_positive = 0
    for et in event_count:
        print('In {} docs: num# {}={}'.format(dataset_prefix, et, event_count[et]))
        count = event_count[et]
        pair_true_positive += (count * (count - 1)) / 2
    return pair_true_positive


# +1
def evaluate_f1_binary(prediction, label, examples, pred_threshold, num_true_positive=None, class_label='OVERALL', target_types=None):
    """
    Given a binary task, we will always assume 0 class index is negative, 1 is positive

    :type examples: list[nlplingo.tasks.eventpair.example.EventPairExample]
    :type num_true_positive: int
    :type target_types: set[str]    # a set of event types that we want to evaluate on
    :rtype: nlplingo.common.utils.F1Score
    """
    ret = []

    #R_dict = defaultdict(int)
    #P_dict = defaultdict(int)
    #C_dict = defaultdict(int)

    print('In eventpair.run.evaluate_f1_binary: #prediction={} #label={} #examples={}'.format(len(prediction), len(label), len(examples)))

    num_correct = 0
    num_true = 0
    num_predict = 0
    for i in range(len(prediction)):
        if target_types is None or (examples[i].eg1.event_type in target_types or examples[i].eg2.event_type in target_types):
            if prediction[i] >= pred_threshold:
                #P_dict[class_labels[i]] += 1
                num_predict += 1
                #print_pair_example(examples[i], class_labels[i], prediction[i])
                if label[i] == 1:
                    num_correct += 1
                    #C_dict[class_labels[i]] += 1

    for i in range(len(label)):
        #if examples[i].eg1.event_type.startswith('Business'):
        #    print('eg1.event_type={}, eg2.event_type={}'.format(examples[i].eg1.event_type, examples[i].eg2.event_type))
        #    print('label[i]={}'.format(label[i]))
        if label[i] == 1 and (target_types is None or examples[i].eg1.event_type in target_types):
            num_true += 1
            #R_dict[class_labels[i]] += 1
            #print_pair_example(examples[i], class_labels[i], prediction[i])

    if num_true_positive is not None:
        print('num_true={} num_true_positive={}'.format(num_true, num_true_positive))
        ret.append(F1Score(num_correct, num_true_positive, num_predict, class_label))
    else:
        print('num_true={} num_true_positive={}'.format(num_true, num_true))
        ret.append(F1Score(num_correct, num_true, num_predict, class_label))
    return ret

def mAP(y_true, y_pred):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_pred)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_eventpair, test_eventpair
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)

    load_extractor_models_from_file = False
    if args.mode in {'test_eventpair'}:
        load_extractor_models_from_file = True

    trigger_extractors = []
    eventpair_extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractor = Extractor(params, extractor_params, embeddings, load_extractor_models_from_file)
        if extractor.model_type.startswith('event-trigger_'):
            trigger_extractors.append(extractor)
        elif extractor.model_type.startswith('event-pair_'):
            eventpair_extractors.append(extractor)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(extractor.model_type))

    if args.mode == 'train_eventpair':
        train_eventpair(params, embeddings, trigger_extractors[0], eventpair_extractors[0], params['pred_threshold'])
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))
