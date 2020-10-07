import codecs
from collections import defaultdict
import logging
import os

import numpy as np
from nlplingo.common.utils import F1Score
from nlplingo.tasks.eventargument.example import EventArgumentExample
from nlplingo.tasks.eventrelation.example import EventEventRelationExample
from nlplingo.tasks.entityrelation.example import EntityRelationExample

logger = logging.getLogger(__name__)


def evaluate_f1(prediction, label, none_class_index, num_true=None):
    """We will input num_true if we are using predicted triggers to score arguments

    ('- prediction=', array([[ 0.00910971,  0.00806234,  0.03608446,  0.94674349],
       [ 0.02211222,  0.01518068,  0.17702729,  0.78567982],
       [ 0.01333893,  0.00946771,  0.03522802,  0.94196534],
       ...,
       [ 0.00706887,  0.01225629,  0.01827211,  0.9624027 ],
       [ 0.0132369 ,  0.03686138,  0.02967645,  0.92022526],
       [ 0.01413057,  0.03428967,  0.02316411,  0.92841566]], dtype=float32))
    ('- label=', array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       ...,
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 1]], dtype=int32))

    label: matrix of size (#instance, #label-types)
    So doing an argmax along 2nd dimension amounts to
    extracting the index of the true/predicted label, for each instance

    ('- label_arg_max=', array([3, 2, 3, ..., 3, 3, 3])

    :type prediction: numpy.matrix
    :type label: numpy.array
    :type none_class_index: int
    :type num_true: int

    Returns:
        nlplingo.common.utils.F1Score, dict
    """

    num_instances = label.shape[0]
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    return evaluate_f1_lists(pred_arg_max, label_arg_max, none_class_index, num_true)


def print_confusion_matrix_for_event_types(confusion_counts, label_indices, domain):
    """
    :type domain: nlplingo.tasks.event_domain.EventDomain
    """
    label_strings = [domain.get_event_type_from_index(i) for i in sorted(label_indices)]
    print(''.join('%20s' % (s) for s in [''] + label_strings))

    for i in sorted(label_indices):
        gold_label = domain.get_event_type_from_index(i)
        a = []
        a.append('%20s' % (gold_label))
        for j in sorted(label_indices):
            predicted_label = domain.get_event_type_from_index(j)
            c = confusion_counts[(i, j)]
            a.append('%20s' % (str(c)))
        print(''.join(a))


def calculate_confusion_matrix(pred_arg_max, label_arg_max):

    label_indices = set()
    for i in pred_arg_max:
        label_indices.add(i)
    for i in label_arg_max:
        label_indices.add(i)

    confusion_counts = defaultdict(int)
    for i in sorted(label_indices):
        for j in sorted(label_indices):
            confusion_counts[(i, j)] = 0

    for i, j in zip(label_arg_max, pred_arg_max):
        confusion_counts[(i, j)] += 1

    return confusion_counts, label_indices


def evaluate_f1_lists(pred_arg_max, label_arg_max, none_class_index, num_true=None):
    """We will input num_true if we are using predicted triggers to score arguments

    :type pred_arg_max: list
    :type label_arg_max: list
    :type none_class_index: int
    :type num_true: int

    Returns:
        nlplingo.common.utils.F1Score, dict
    """

    num_instances = label_arg_max.shape[0]

    # check whether each element in label_arg_max != none_class_index
    # So the result is a 1-dim vector of size #instances, where each element is True or False
    # And then we sum up the number of True elements to obtain the num# of true events
    if num_true is None:
        num_true = np.sum(label_arg_max != none_class_index)
    num_predict = np.sum(pred_arg_max != none_class_index)

    c = 0
    for i, j in zip(label_arg_max, pred_arg_max):
        if i == j and i != none_class_index:
            c += 1

    # calculate F1 for each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    for i in range(len(label_arg_max)):
        if label_arg_max[i] != none_class_index:
            recall_counts[label_arg_max[i]] += 1
            if pred_arg_max[i] == label_arg_max[i]:
                correct_counts[label_arg_max[i]] += 1
    for i in range(len(pred_arg_max)):
        if pred_arg_max[i] != none_class_index:
            precision_counts[pred_arg_max[i]] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown


def evaluate_arg_f1_lists(event_domain, gold_data, predict_data):

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    c = len(gold_data.intersection(predict_data))
    num_true = len(gold_data)
    num_predict = len(predict_data)

    for d in gold_data:
        label = d[1][1]
        recall_counts[event_domain.get_event_role_index(label)] += 1

    for d in predict_data:
        label = d[1][1]
        index = event_domain.get_event_role_index(label)
        precision_counts[index] += 1
        if d in gold_data:
            correct_counts[index] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown, gold_data

def evaluate_eer_f1_lists(event_domain, gold_data, predict_data):

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    c = len(gold_data.intersection(predict_data))
    num_true = len(gold_data)
    num_predict = len(predict_data)

    for d in gold_data:
        # TODO: it seems by making d[1][1] the label, we are expecting an ordering in constructing d[1]
        print ("DBG: " + str(d))
        # label = d[1][1]
        label = d[1]
        recall_counts[event_domain.get_eer_type_index(label)] += 1

    for d in predict_data:
        # label = d[1][1]
        label = d[1]
        index = event_domain.get_eer_type_index(label)
        precision_counts[index] += 1
        if d in gold_data:
            correct_counts[index] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown, gold_data

def eer_score_breakdown(event_domain, gold, predict, neg, unfold_tuple=True):
    correct = 0
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    total = len(predict)

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for i in range(total):
        if unfold_tuple:
            golden = gold[i][0]
        else:
            golden = gold[i]
        # print('golden', golden)
        prediction = predict[i]
        # print('prediction', prediction)
        recall_counts[golden] += 1
        precision_counts[prediction] += 1
        if golden == prediction:
            correct += 1
            correct_counts[prediction] += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if prediction != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)
    print_score_breakdown_from_domain(event_domain, score_breakdown)
    return result

# TODO: unify
def arg_score_breakdown(event_domain, gold, predict, neg, unfold_tuple=True):
    correct = 0
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    total = len(predict)

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for i in range(total):
        if unfold_tuple:
            golden = gold[i][0]
        else:
            golden = gold[i]
        # print('golden', golden)
        prediction = predict[i]
        # print('prediction', prediction)
        recall_counts[golden] += 1
        precision_counts[prediction] += 1
        if golden == prediction:
            correct += 1
            correct_counts[prediction] += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if prediction != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)
    print_arg_score_breakdown_from_domain(event_domain, score_breakdown)
    return result

# TODO: unify
def entity_relation_score_breakdown(event_domain, gold, predict, neg, unfold_tuple=True):
    correct = 0
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    total = len(predict)

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for i in range(total):
        if unfold_tuple:
            golden = gold[i][0]
        else:
            golden = gold[i]
        # print('golden', golden)
        prediction = predict[i]
        # print('prediction', prediction)
        recall_counts[golden] += 1
        precision_counts[prediction] += 1
        if golden == prediction:
            correct += 1
            correct_counts[prediction] += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if prediction != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)
    print_entity_relation_score_breakdown_from_domain(event_domain, score_breakdown)
    return result

def evaluate_entity_relation_f1_lists(event_domain, gold_data, predict_data):

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    c = len(gold_data.intersection(predict_data))
    num_true = len(gold_data)
    num_predict = len(predict_data)

    for d in gold_data:
        # TODO: it seems by making d[1][1] the label, we are expecting an ordering in constructing d[1]
        print ("DBG: " + str(d))
        # label = d[1][1]
        label = d[1]
        recall_counts[event_domain.get_entity_relation_type_index(label)] += 1

    for d in predict_data:
        # label = d[1][1]
        label = d[1]
        index = event_domain.get_entity_relation_type_index(label)
        precision_counts[index] += 1
        if d in gold_data:
            correct_counts[index] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown, gold_data


def evaluate_arg_f1(event_domain, test_label, test_examples, predictions, scoring_domain=None,
                    gold_labels=None):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type test_label: np.array
    :type test_examples: list[nlplingo.tasks.eventargument.EventArgumentExample]
    :type scoring_domain: nlplingo.event.event_domain.EventDomain

    Returns:
        common.utils.F1Score
    """
    assert len(test_label) == len(test_examples)
    assert len(predictions) == len(test_examples)

    none_class_index = event_domain.get_event_role_index('None')

    if gold_labels is not None:
        gold_data = gold_labels
    else:
        gold_data = set()
        test_arg_max = np.argmax(test_label, axis=1)
        for i, index in enumerate(test_arg_max):
            if index != none_class_index:
                eg = test_examples[i]
                """:type: nlplingo.tasks.eventargument.EventArgumentExample"""
                if (
                    (scoring_domain is not None and scoring_domain.event_type_in_domain(eg.anchor.label)) or
                    scoring_domain is None
                ):
                    id_ = (
                        eg.arg0.sentence.docid,
                        eg.arg1.span.head().start_char_offset(),
                        eg.arg1.span.head().end_char_offset()
                    )
                    if isinstance(eg, EventArgumentExample):
                        label = (
                            eg.arg0.span.label,
                            event_domain.get_event_role_from_index(index)
                        )
                    else:
                        raise RuntimeError('test_example not an instance of an implemented type.')
                    gold_data.add((id_, label))
                    # tabulating for score_breakdown
                    #recall_counts[index] += 1

    predict_data = set()
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            # pred_scores = predictions[i]
            # score_strings = []
            # for j, score in enumerate(pred_scores):
            #     if score >= 0.5:
            #         score_strings.append('{}:{}'.format(str(j), '%.2f' % score))
            #print('{}: {}'.format(index, ', '.join(score_strings)))

            eg = test_examples[i]
            """:type: nlplingo.tasks.eventargument.EventArgumentExample"""
            if (
                (scoring_domain is not None and scoring_domain.event_type_in_domain(eg.anchor.label)) or
                scoring_domain is None
            ):
                id_ = (
                    eg.arg0.sentence.docid,
                    eg.arg1.span.head().start_char_offset(),
                    eg.arg1.span.head().end_char_offset()
                )
                if isinstance(eg, EventArgumentExample):
                    label = (
                        eg.arg0.span.label,
                        event_domain.get_event_role_from_index(index)
                    )
                else:
                    raise RuntimeError('test_example not an instance of an implemented type.')
                predict_data.add((id_, label))

    # predict_data = set()
    # for i in range(len(predictions)):
    #     scores = predictions[i]
    #     for index in range(len(scores)):
    #         score = scores[index]
    #         if score >= 0.5 and index != none_class_index:
    #             eg = test_examples[i]
    #             """:type: nlplingo.tasks.eventargument.EventArgumentExample"""
    #             if (scoring_domain is not None and scoring_domain.event_type_in_domain(
    #                     eg.anchor.label)) or scoring_domain is None:
    #                 id = '{}_{}_{}'.format(eg.sentence.docid,
    #                                        eg.argument.head().start_char_offset(),
    #                                        eg.argument.head().end_char_offset())
    #                 if isinstance(eg, EventArgumentExample):
    #                     label = '{}_{}'.format(eg.anchor.label,
    #                                            event_domain.get_event_role_from_index(index))
    #                 elif isinstance(eg, ArgumentExample):
    #                     label = '{}_{}'.format(eg.event_type,
    #                                            event_domain.get_event_role_from_index(index))
    #                 predict_data.add('{}__{}'.format(id, label))
    return evaluate_arg_f1_lists(event_domain, gold_data, predict_data)


def evaluate_eer_f1(event_domain, test_label, test_examples, predictions, scoring_domain=None,
                    gold_labels=None):
    assert len(test_label) == len(test_examples)
    assert len(predictions) == len(test_examples)

    none_class_index = event_domain.get_eer_type_index('NA')

    if gold_labels is not None:
        gold_data = gold_labels
    else:
        gold_data = set()
        test_arg_max = np.argmax(test_label, axis=1)
        for i, index in enumerate(test_arg_max):
            if index != none_class_index:
                eg = test_examples[i]
                """:type: nlplingo.eventrelation.eventrelation.EventEventRelationExample"""
                if (
                    (scoring_domain is not None and scoring_domain.eer_type_in_domain(eg.eer_type)) or
                    scoring_domain is None
                ):
                    id_ = (
                        eg.sentence.docid,
                        eg.arg0.span.start_char_offset(),
                        eg.arg0.span.end_char_offset(),
                        eg.arg1.span.start_char_offset(),
                        eg.arg1.span.end_char_offset()
                    )
                    if isinstance(eg, EventEventRelationExample):
                        label = (
                            event_domain.get_eer_type_from_index(index)
                        )
                    else:
                        raise RuntimeError('test_example not an instance of an implemented type.')
                    gold_data.add((id_, label))
                    # tabulating for score_breakdown
                    #recall_counts[index] += 1

    predict_data = set()
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            # pred_scores = predictions[i]
            # score_strings = []
            # for j, score in enumerate(pred_scores):
            #     if score >= 0.5:
            #         score_strings.append('{}:{}'.format(str(j), '%.2f' % score))
            #print('{}: {}'.format(index, ', '.join(score_strings)))

            eg = test_examples[i]
            """:type: nlplingo.eventrelation.eventrelation.EventEventRelationExample"""
            if (
                (scoring_domain is not None and scoring_domain.eer_type_in_domain(eg.eer_type)) or
                scoring_domain is None
            ):
                id_ = (
                    eg.sentence.docid,
                    eg.arg0.span.start_char_offset(),
                    eg.arg0.span.end_char_offset(),
                    eg.arg1.span.start_char_offset(),
                    eg.arg1.span.end_char_offset()
                )
                if isinstance(eg, EventEventRelationExample):
                    label = (
                        event_domain.get_eer_type_from_index(index)
                    )
                else:
                    raise RuntimeError('test_example not an instance of an implemented type.')
                predict_data.add((id_, label))

    return evaluate_eer_f1_lists(event_domain, gold_data, predict_data)


def evaluate_entity_relation_f1(event_domain, test_label, test_examples, predictions, scoring_domain=None,
                    gold_labels=None):
    assert len(test_label) == len(test_examples)
    assert len(predictions) == len(test_examples)

    none_class_index = event_domain.get_entity_relation_type_index('None')

    if gold_labels is not None:
        gold_data = gold_labels
    else:
        gold_data = set()
        test_arg_max = np.argmax(test_label, axis=1)
        for i, index in enumerate(test_arg_max):
            if index != none_class_index:
                eg = test_examples[i]
                if (
                    (scoring_domain is not None and scoring_domain.entity_relation_type_in_domain(eg.relation_type)) or
                    scoring_domain is None
                ):
                    id_ = (
                        eg.arg0.sentence.docid,
                        eg.arg0.span.start_char_offset(),
                        eg.arg0.span.end_char_offset(),
                        eg.arg1.span.start_char_offset(),
                        eg.arg1.span.end_char_offset()
                    )
                    if isinstance(eg, EntityRelationExample):
                        label = (
                            event_domain.get_entity_relation_type_from_index(index)
                        )
                    else:
                        raise RuntimeError('test_example not an instance of an implemented type.')
                    gold_data.add((id_, label))
                    # tabulating for score_breakdown
                    #recall_counts[index] += 1

    predict_data = set()
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            # pred_scores = predictions[i]
            # score_strings = []
            # for j, score in enumerate(pred_scores):
            #     if score >= 0.5:
            #         score_strings.append('{}:{}'.format(str(j), '%.2f' % score))
            #print('{}: {}'.format(index, ', '.join(score_strings)))

            eg = test_examples[i]
            if (
                (scoring_domain is not None and scoring_domain.entity_relation_type_in_domain(eg.relation_type)) or
                scoring_domain is None
            ):
                id_ = (
                    eg.arg0.sentence.docid,
                    eg.arg0.span.start_char_offset(),
                    eg.arg0.span.end_char_offset(),
                    eg.arg1.span.start_char_offset(),
                    eg.arg1.span.end_char_offset()
                )
                if isinstance(eg, EntityRelationExample):
                    label = (
                        event_domain.get_entity_relation_type_from_index(index)
                    )
                else:
                    raise RuntimeError('test_example not an instance of an implemented type.')
                predict_data.add((id_, label))

    return evaluate_entity_relation_f1_lists(event_domain, gold_data, predict_data)


def print_score_breakdown(extractor, score_breakdown):
    """
    :type extractor: nlplingo.nn.extractor.Extractor
    :type score_breakdown: dict
    """
    for index, f1_score in score_breakdown.items():
        et = extractor.domain.get_event_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

def print_score_breakdown_from_domain(domain, score_breakdown):
    """
    :type extractor: nlplingo.nn.tasks.event_domain.EventDomain
    :type score_breakdown: dict
    """
    for index, f1_score in score_breakdown.items():
        et = domain.get_eer_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

def print_arg_score_breakdown_from_domain(domain, score_breakdown):
    """
    :type extractor: nlplingo.nn.tasks.event_domain.EventDomain
    :type score_breakdown: dict
    """
    for index, f1_score in score_breakdown.items():
        et = domain.get_event_role_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

def print_entity_relation_score_breakdown_from_domain(domain, score_breakdown):
    """
    :type extractor: nlplingo.nn.tasks.event_domain.EventDomain
    :type score_breakdown: dict
    """
    for index, f1_score in score_breakdown.items():
        et = domain.get_entity_relation_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

def write_score_to_file(extractor, score, score_breakdown, filepath):
    """
    :type extractor: nlplingo.nn.extractor.Extractor
    :type score: nlplingo.common.utils.F1Score
    :type score_breakdown: dict
    :type filepath: str
    """
    with codecs.open(filepath, 'w', encoding='utf-8') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            et = extractor.domain.get_event_type_from_index(index)
            f.write('{}\t{}\n'.format(et, f1_score.to_string()))


def report_top_n_sample_info(n, predictions, examples, labels, thresholds,
                             extractor, suffix=''):
    """
    Report information on top N false positives and top N samples, when sorting
    by prediction score, for each label
    :param n: number of samples to report on
    :param predictions: prediction array
    :param examples: examples list
    :param labels: true labels array
    :param thresholds: thresholds array -- may be None
    :param extractor: extractor (used to get domain and write path
    :param suffix: string to append to the write path
    """
    out_directory = os.path.dirname(extractor.class_thresholds_path)
    if not os.path.isdir(out_directory):
        raise IOError("No such directory: {}".format(out_directory))
    write_top_n_incorrect_samples(
        n, predictions, examples, labels, thresholds, extractor, suffix=suffix)
    write_top_n_samples(
        n, predictions, examples, labels, thresholds, extractor, suffix=suffix)


def write_top_n_incorrect_samples(n, predictions, examples, label_matrix,
                                  thresholds, extractor, suffix=''):
    logger.debug('Prediction shape = {}'.format(predictions.shape))
    logger.debug('Examples length = {}'.format(len(examples)))
    out_name = extractor.class_thresholds_path + '.top_{}_false_positive'.format(n)
    if suffix:
        out_name += '_' + suffix
    domain = extractor.domain
    with codecs.open(out_name, 'w', encoding='utf8') as f:
        for label_index in range(label_matrix.shape[1]):
            f.write('### Class: {} ({}) ###\n'.format(
                domain.get_event_type_from_index(label_index), label_index))
            p_e = [(p, e) for p, e in
                   zip(predictions[:, label_index], examples)]
            sorted_p_e = sorted(p_e, key=lambda x: x[0],
                                reverse=True)  # sort in descending order
            try:
                threshold = thresholds[label_index]
            except TypeError:
                threshold = 0.5

            false_positives = 0
            for rank, (pred, ex) in enumerate(sorted_p_e):
                true_class_idx = domain.get_event_type_index(ex.event_type)
                if pred < threshold:
                    f.write('##### threshold reached, remainder are all '
                            'negatives ####\n')
                    break
                elif false_positives >= n:
                    f.write('##### top n reached ####\n')
                    break
                elif true_class_idx == label_index:
                    if true_class_idx == domain.get_event_type_index('None'):
                        f.write('Rank: {} is a true negative\n'.format(
                            rank + 1))
                    else:
                        f.write('Rank: {} is a true positive for {}\n'.format(
                            rank + 1,
                            domain.get_event_type_from_index(label_index)))
                elif true_class_idx != label_index:
                    if true_class_idx == domain.get_event_type_index('None'):
                        f.write('Rank: {} is a false positive for {} (or a '
                                'missed annotation)\n'.format(
                            rank + 1,
                            domain.get_event_type_from_index(label_index)))
                    elif label_index == domain.get_event_type_index('None'):
                        f.write('Rank: {} is a false negative for {}\n'.format(
                            rank + 1,
                            domain.get_event_type_from_index(label_index)))
                    else:
                        f.write('Rank: {} is a false positive for {}\n'.format(
                            rank + 1,
                            domain.get_event_type_from_index(label_index)))
                    false_positives += 1

                    f.write('\tScore: {}\tThreshold: {}\tTrue class: {} ({})\n'
                            ''.format(pred, threshold, ex.event_type,
                                      true_class_idx))
                    f.write(u'\tTrigger: {}\n'.format(ex.anchor.to_string()))
                    f.write(u'\tSentence: {}\n'.format(ex.sentence.to_string()))

                f.write('------\n')


def write_top_n_samples(n, predictions, examples, label_matrix, thresholds,
                        extractor, suffix=''):
    logger.debug('Prediction shape = {}'.format(predictions.shape))
    logger.debug('Examples length = {}'.format(len(examples)))
    out_name = extractor.class_thresholds_path + '.top_{}_predictions'.format(n)
    if suffix:
        out_name += '_' + suffix
    domain = extractor.domain
    with codecs.open(out_name, 'w', encoding='utf8') as f:
        for label_index in range(label_matrix.shape[1]):
            p_e = [(p, e) for p, e in
                   zip(predictions[:, label_index], examples)]
            sorted_p_e = sorted(p_e, key=lambda x: x[0],
                                reverse=True)  # sort in descending order
            try:
                threshold = thresholds[label_index]
            except TypeError:
                threshold = 0.5

            f.write('### Class: {} ({}) ###\n### Threshold: {} ###\n'.format(
                label_index,
                domain.get_event_type_from_index(label_index),
                threshold))

            if label_index == domain.get_event_type_index("None"):
                f.write("# Not currently generating examples for NA class.\n")
                continue

            for rank, (pred, ex) in enumerate(sorted_p_e):
                gold_class_idx = domain.get_event_type_index(ex.event_type)
                gold_is_na = gold_class_idx == domain.get_event_type_index("None")
                gold_is_current_label = gold_class_idx == label_index
                current_label_is_predicted = pred >= threshold

                if rank >= n:
                    f.write('##### top {} reached ####\n'.format(n))
                    break

                f.write('Rank: {}\tScore: {}\tGold: {} ({})\n'.format(
                    rank + 1, pred, gold_class_idx, ex.event_type))
                prediction_message = ' {}\n'.format(
                    domain.get_event_type_from_index(label_index))
                if current_label_is_predicted:
                    prediction_message = 'Predicted' + prediction_message
                else:
                    prediction_message = 'Did not predict' + prediction_message
                f.write(prediction_message)

                if gold_is_current_label:  # TP or FN
                    if current_label_is_predicted:  # TP
                        f.write('Correct (TP)\n')
                    else:  # FN
                        f.write('Incorrect (FN)\n')
                else:  # FP or TN
                    if current_label_is_predicted:  # FP
                        if gold_is_na:
                            f.write('Incorrect (FP) - Not Annotated or NA\n')
                        else:
                            f.write('Incorrect (FP)')
                    else:  # TN
                        f.write('Correct (TN)\n')

                f.write(u'Trigger: {}\n'.format(ex.anchor.to_string()))
                f.write(u'Sentence: {}\n'.format(ex.sentence.to_string()))
                f.write('-------------\n')


def calculate_all_thresholds(predictions, labels, domain):
    C, R, P = 0, 0, 0
    best_thresholds = np.zeros(labels.shape[1])
    for label_index in range(labels.shape[1]):
        et = domain.get_event_type_from_index(label_index)
        threshold, max_f1_score = calculate_threshold_for_class(
            predictions, labels, label_index)
        C += max_f1_score.c
        R += max_f1_score.num_true
        P += max_f1_score.num_predict
        max_f1_score.class_label = et
        logger.info('Per-class threshold tuning: {}'.format(
            max_f1_score.to_string()))
        best_thresholds[label_index] = threshold
    logger.info('Per-class threshold tuning: {}'.format(
        F1Score(C, R, P, class_label='OVERALL').to_string()))
    return best_thresholds


def calculate_threshold_for_class(prediction, label, label_index):
    """
    Use a Precision-Recall Curve (PRC) to find the threshold at which the
    highest F1 value occurs for a given label.
    :type prediction: numpy.ndarray
    :type label: numpy.ndarray
    :type label_index: int
    :rtype: numpy.float32, nlplingo.common.utils.F1Score
    """
    default_threshold = 0.5
    p_l = [(p, l) for p, l in zip(prediction[:, label_index], label[:, label_index])]
    sorted_p_l = sorted(p_l, key=lambda x: x[0], reverse=True)  # sort in descending order
    sorted_p = [p for p, l in sorted_p_l]   # sorted descending probability
    sorted_l = [l for p, l in sorted_p_l]   # label sorted according to descending probability
    cum_correct = np.cumsum(sorted_l)
    R = np.sum(sorted_l)  # number of true instances of this label
    if R == 0:
        logger.info('WARNING: there are no true positives for label {}!'.format(
            label_index))

    # Find threshold with highest F1
    max_f1_score = F1Score(0, 0, 0)
    threshold = default_threshold
    for i, prob in enumerate(sorted_p):
        C = cum_correct[i]  # number of correct predictions at this point in PRC
        P = i + 1  # number of predictions of this label at this point in curve
        f1_score = F1Score(C, R, P)
        # using > and not >= may favor precision over recall very rarely
        if f1_score.f1 > max_f1_score.f1:
            max_f1_score = f1_score
            threshold = prob

    if threshold == 0:  # occurs when threshold rounds down to 0
        logger.info('WARNING: threshold={} being replaced with default ({})'
                    .format(threshold, default_threshold))
        threshold = default_threshold

    logger.info('label_index={} threshold={}'.format(label_index, threshold))

    return threshold, max_f1_score


def evaluate_baseline_and_best_multi_label_f1s(preds, labels, extractor,
                                               score_prefix):
    """
    Evaluate multi-label model predictions.  First get an untuned baseline, then
    calculate oracle thresholds and show the best possible score.
    :type preds: np.ndarray
    :type labels: np.ndarray
    :type extractor: nlplingo.nn.extractor.Extractor
    :type score_prefix: str
    """
    none_idx = extractor.domain.get_event_type_index('None')

    # Get untuned baseline score
    logger.info('Multi-label score using untuned thresholds')
    score, score_breakdown = evaluate_multi_label_f1(
        preds, labels, none_idx, thresholds=None)
    logger.info(score.to_string())
    print_score_breakdown(extractor, score_breakdown)
    write_score_to_file(
        extractor, score, score_breakdown, score_prefix)

    # Get tuned (oracle) thresholds
    logger.info('Tuning thresholds...')
    best_thresholds = calculate_all_thresholds(preds, labels, extractor.domain)

    # Evaluate again
    logger.info('Multi-label score using oracle thresholds')
    score, score_breakdown = evaluate_multi_label_f1(
        preds, labels, none_idx, thresholds=best_thresholds)
    logger.info(score.to_string())
    print_score_breakdown(extractor, score_breakdown)
    write_score_to_file(
        extractor, score, score_breakdown, score_prefix + '.oracle')

    # return best thresholds for saving/reuse
    return best_thresholds


def evaluate_multi_label_f1(prediction, label, none_class_idx, thresholds=None):
    """
    :param prediction:
    :param label:
    :param none_class_idx:
    :param thresholds: thresholds for each class at or above which a positive classification is made.
                        Should be an ndarray of float-likes of shape (1, #classes), or None.
    :return: overall score and per-class score breakdown
    """

    # get the true positives
    label = label.astype(u'int32')
    non_null_labels = np.copy(label)
    non_null_labels[:, none_class_idx] = 0
    number_true = np.sum(non_null_labels)
    print('true:{}'.format(number_true))

    # get the predicted positives
    non_null_predictions = np.copy(prediction)
    non_null_predictions[:, none_class_idx] = 0
    # convert per-class binary probability distributions to boolean feature vec
    if thresholds is None:
        thresholds = 0.5
    non_null_predictions = non_null_predictions >= thresholds
    non_null_predictions = non_null_predictions.astype(u'int32')    # converts True False into 1 0
    number_predicted = np.sum(non_null_predictions)
    print('pred:{}'.format(number_predicted))

    # get the intersection of true and predicted positives
    # Note: true negatives are not included here
    correct_positive_predictions = non_null_labels & non_null_predictions
    number_correct = np.sum(correct_positive_predictions)
    print('cor:{}'.format(number_correct))

    # per-class scores
    score_breakdown = {}
    for class_label in range(non_null_labels.shape[1]):
        if class_label != none_class_idx:
            number_true_for_class = np.sum(non_null_labels[:, class_label])
            number_pred_for_class = np.sum(non_null_predictions[:, class_label])
            correct_for_class = (non_null_labels[:, class_label] &
                                 non_null_predictions[:, class_label])
            number_correct_for_class = np.sum(correct_for_class)
            score_breakdown[class_label] = F1Score(number_correct_for_class,
                                                   number_true_for_class,
                                                   number_pred_for_class)

    # return an F1Score and a dict of F1Scores
    return (F1Score(number_correct, number_true, number_predicted),
            score_breakdown)
