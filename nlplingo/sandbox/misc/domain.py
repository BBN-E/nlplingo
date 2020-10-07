
"train" => "/d4m/ears/expts/46178crp_1A_ana1+2.web_24K-12K.giza_mt_w_seeds_rank1.sst.lid_0.008/expts/create_corpus_from_sgm/filelist.txt"
"dev" = > "/d4m/ears/expts/46159crp_1A_ana1+2.sst/expts/create_corpus_from_sgm/filelist_all_no_cs_2.txt"
"test" = > "/d4m/ears/expts/46159crp_1A_ana1+2.sst/expts/create_corpus_from_sgm/filelist_all_no_cs.txt"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import argparse
import os
from collections import defaultdict

import numpy as np
from future.builtins import range
from keras.models import load_model as keras_load_model
from nlplingo.common.utils import F1Score
from nlplingo.common.utils import IntPair
from nlplingo.embeddings.word_embeddings import WordEmbedding
from nlplingo.tasks.event_domain import EventDomain
from nlplingo.sandbox.common.parameters import Parameters
from nlplingo.sandbox.model import MaxPoolEmbeddedWordPairModel
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import Token
from pycube.utils.bbn_text_segment import BBNSegmentReader


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

    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type none_label_index: int

    Returns:
        nlplingo.common.utils.F1Score
    """

    num_instances = label.shape[0]
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)

    #print('- prediction=', prediction)
    #print('- label=', label)
    #print('- none_class_index=', none_class_index)
    #print('- label_arg_max=', label_arg_max)
    #print('- pred_arg_max=', pred_arg_max)

    # check whether each element in label_arg_max != none_class_index
    # So the result is a 1-dim vector of size #instances, where each element is True or False
    # And then we sum up the number of True elements to obtain the num# of true events
    if num_true == None:
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


class DomainGenerator(object):
    def __init__(self, params, word_embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        """
        self.params = params
        self.word_embeddings = word_embeddings
        self.max_sent_length = params.get_int('max_sent_length')
        self.embedding_prefix = params.get_string('embedding.prefix')
        print(self.embedding_prefix)
        self.statistics = defaultdict(int)

    def to_sentence(self, sentence_text, docid, sentence_id):
        """
        :type sentence_text: str
        :type docid: str
        :type sentence_id: str
        :rtype: nlplingo.text.text_span.Sentence
        """
        sentence_length_thus_far = 0
        tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""
        for i, token_string in enumerate(sentence_text.split(' ')):
            start = sentence_length_thus_far
            end = start + len(token_string)
            token = Token(IntPair(start, end), i, token_string, lemma=None, pos_tag=None)
            sentence_length_thus_far += len(token_string) + 1  # +1 for space
            tokens.append(token)
        sent = Sentence(docid, IntPair(tokens[0].start_char_offset(), tokens[-1].end_char_offset()), sentence_text, tokens, 0)
        sent.sent_id = sentence_id
        return sent

    def generate_examples(self, segments):
        ret = []
        """:type: list[DomainExample]"""

        for segment in segments:
            docid = segment.field_value['DOCUMENT_ID']
            guid = segment.field_value['GUID']
            sentence_text = segment.field_value['GIZA_MT_0']
            label_string = segment.field_value['SEARCH_GIZA_MT_0']

            sentence = self.to_sentence(sentence_text, docid, guid)

            label_tokens = label_string.split()
            if len(label_tokens) == 0:
                label_tokens.append('None')

            for label in label_tokens:
                example = DomainExample(sentence, params, label)
                self._generate_example(example, [None] + sentence.tokens)
                ret.append(example)

        return ret

    # @staticmethod
    # def _docid_from_guid(guid):
    #     tokens = guid[1:-1].replace('[', '').replace(']', ' ').split(' ')
    #     return '_'.join(tokens[0:-1])
    #
    # def _use_sentence(self, sentence):
    #     if sentence.number_of_tokens() < 1:
    #         return False
    #     if sentence.number_of_tokens() >= self.max_sent_length:
    #         self.statistics['Skipping overly long sentence'] += 1
    #         return False
    #     return True

    @classmethod
    def _generate_example(cls, example, tokens):
        """
        :type example: nlplingo.tasks.wordpair.DomainExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        cls.assign_vector_data(tokens, example)

    @staticmethod
    def assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.tasks.wordpair.DomainExample
        """
        for i, token in enumerate(tokens):
            if token and token.has_vector:
                example.vector_data[i] = token.vector_index

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.tasks.domain.DomainExample]
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec'].append(example.vector_data)
            data_dict['label'].append(example.label)
        return data_dict

class DomainExample(object):
    def __init__(self, sentence, params, label):
        """We are given a token, sentence as context, and event_type (present during training)
        :type sentence: nlplingo.text.text_span.Sentence
        :type params: nlplingo.common.parameters.Parameters
        :type label_string: str
        """
        self.sentence = sentence
        self.label = label
        self._allocate_arrays(params.get_int('max_sent_length'),
                              params.get_int('embedding.none_token_index'), params.get_string('cnn.int_type'))

    def _allocate_arrays(self, max_sent_length, none_token_index, int_type):
        """Allocates feature vectors and matrices for examples from this sentence
        :type max_sent_length: int
        :type none_token_index: int
        :type int_type: str
        """
        self.vector_data = none_token_index * np.ones(max_sent_length, dtype=int_type)

def generate_wordpair_data_feature(generator, examples):
    """
    :type generator: nlplingo.tasks.wordpair.DomainGenerator
    :type examples: list[DomainExample]
    """
    data = generator.examples_to_data_dict(examples)

    data_list = [np.asarray(data['word_vec'])]
    label = np.asarray(data['label'])

    print('#domain-examples=%d' % (len(examples)))
    print('data word_vec.len=%d label.len=%d' % (len(data['word_vec']), len(data['label'])))

    for key in generator.statistics:
        print('{} = {}'.format(key, generator.statistics[key]))

    return (examples, data, data_list, label)


def train(event_domain, params, word_embeddings, generator, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type generator: nlplingo.tasks.wordpair.DomainGenerator
    :type examples: list[DomainExample]
    """
    (train_examples, train_data, train_data_list, train_label) = generate_wordpair_data_feature(generator, examples)

    wp_model = MaxPoolEmbeddedWordPairModel(params, word_embeddings, event_domain)
    wp_model.fit(train_data_list, train_label)
    print('==== Saving model ====')
    wp_model.save_keras_model(os.path.join(params.get_string('model_dir'), 'domain.hdf'))

def load_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'domain.hdf'))
    return model

def test(event_domain, params, word_embeddings, generator, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type generator: nlplingo.tasks.wordpair.DomainGenerator
    :type examples: list[DomainExample]
    """
    print('==== Loading model ====')
    domain_model = load_model(params.get_string('model_dir'))

    (test_examples, test_data, test_data_list, test_label) = generate_wordpair_data_feature(generator, examples)
    predictions = domain_model.predict(test_data_list)
    score = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'))

    print(score.to_string())
    with open(os.path.join(params.get_string('output_dir'), 'test_domain.score'), 'w') as f:
        f.write(score.to_string() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_file')
    parser.add_argument('--mode')
    parser.add_argument('--params')
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    event_domain = EventDomain(['BUS', 'LAW', 'GOV', 'LIF'])

    # load word embeddings
    word_embeddings = WordEmbedding(params, params.get_string('embedding.embedding_file'),
                                    params.get_int('embedding.vocab_size'), params.get_int('embedding.vector_size'))

    generator = DomainGenerator(params, word_embeddings)

    segments = [segment for segment in BBNSegmentReader(args.segment_file)]
    """:type: list[pycube.utils.bbn_text_segment.BBNTextSegment]"""

    examples = generator.generate_examples(segments)
    """:type: list[DomainExample]"""

    if args.mode == 'train_domain':
        train(event_domain, params, word_embeddings, generator, examples)
    elif args.mode == 'test_domain':
        test(event_domain, params, word_embeddings, generator, examples)


