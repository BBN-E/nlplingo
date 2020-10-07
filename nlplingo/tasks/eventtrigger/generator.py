from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import codecs
from collections import defaultdict
import json
import os
import random
import re

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import trigger_pos_category
from nlplingo.common.io_utils import safeprint
from nlplingo.text.text_span import Anchor

from nlplingo.tasks.eventtrigger.example import EventTriggerExample
from nlplingo.tasks.eventtrigger.feature import EventTriggerFeatureGenerator
from nlplingo.tasks.common.examplegenerator import ExampleGenerator

import logging
logger = logging.getLogger(__name__)


class EventKeywordList(object):
    def __init__(self, filepath):
        self.event_type_to_keywords = dict()
        self.keyword_to_event_types = defaultdict(set)
        self._read_keywords(filepath)

    def _read_keywords(self, filepath):
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        for data in datas:
            et_string = data['event_type']
            keywords = set()
            if 'keywords' in data:
                keywords.update(set(data['keywords']))
            if 'variants' in data:
                keywords.update(set(data['variants']))
            if 'hyponym_words' in data:
                keywords.update(set(data['hyponym_words']))
            if 'expanded_keywords' in data:
                keywords.update(set(data['expanded_keywords']))

            self.event_type_to_keywords[et_string] = set(kw.replace(' ', '_') for kw in keywords)
            for kw in keywords:
                self.keyword_to_event_types[kw].add(et_string)

    def get_event_types_for_tokens(self, tokens):
        """practically, we currently only deal with unigrams and bigrams
        :type tokens: list[nlplingo.text.text_span.Token]
        """
        ngrams = tokens[0:2]
        text = '_'.join(token.text for token in ngrams)
        text_with_pos_suffix = text + ngrams[-1].pos_suffix()

        event_types = set()
        if text_with_pos_suffix.lower() in self.keyword_to_event_types:
            event_types = self.keyword_to_event_types[text_with_pos_suffix.lower()]
        elif text.lower() in self.keyword_to_event_types:
            event_types = self.keyword_to_event_types[text.lower()]
        return event_types

    def print_statistics(self):
        # keywords which are associated with multiple event types
        for kw, ets in self.keyword_to_event_types.items():
            if len(ets) > 1:
                print('{} #ets={} {}'.format(kw, len(ets), ', '.join(ets)))


class EventTriggerExampleGenerator(ExampleGenerator):

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class for generating candidate Datapoint objects for the event trigger
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EventTriggerExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)

        self.negative_trigger_words = read_file_to_set(params['trigger.negative_words'])

        # for instance, some of the sentences we have not annotated in precursor, so those contain false negatives
        if 'trigger.sentence_ids_to_discard' in params:
            self.sentence_ids_to_discard = self._read_false_negatives(
                params['trigger.sentence_ids_to_discard'])
        else:
            self.sentence_ids_to_discard = dict()

        if 'trigger.candidate_span_file' in params:
            self.trigger_candidate_span = self._read_candidate_span_file(params['trigger.candidate_span_file'])
            """:type: defaultdict[str, set[IntPair]]"""
        else:
            self.trigger_candidate_span = None

        # if an anchor has this span, we will always generate a trigger example, even if we need to generate 'None' label
        self.only_generate_annotated_spans = False
        if 'trigger.spans_to_generate_examples' in params:
            self.spans_to_generate_examples = self._read_candidate_span_file(
                params['trigger.spans_to_generate_examples'])
            """:type: defaultdict[str, set[IntPair]]"""
            if params.get("trigger.spans_to_generate_examples.only", False):
                self.only_generate_annotated_spans = True
        else:
            self.spans_to_generate_examples = None

        self.restrict_none_examples_using_keywords = params.get('trigger.restrict_none_examples_using_keywords', False)
        if self.restrict_none_examples_using_keywords:
            self.event_keyword_list = EventKeywordList(params['trigger.event_keywords'])

        self.do_not_tag_as_none_dict = self._get_do_not_tag_as_none_dict(params)
        """:type: dict"""

        self.positive_triggers = self._get_positive_trigger_keywords()
        """:type: dict"""

    def _get_positive_trigger_keywords(self):
        keyword_path = self.extractor_params.get('trigger.positive_keywords')
        positive_triggers = None
        if keyword_path is not None:
            if os.path.isfile(keyword_path):
                with codecs.open(keyword_path, 'r', encoding='utf8') as f:
                    positive_triggers = json.load(f)
        return positive_triggers

    def _save_positive_trigger_keywords(self, examples):
        logger.info("Getting positive training triggers from examples")
        keywords_path = self.extractor_params.get('trigger.positive_keywords')

        # get keywords
        keywords = {}
        for example in examples:
            kw = example.get_keyword()
            for event_type in example.event_type:
                if event_type != 'None':
                        keywords.setdefault(event_type, dict()).setdefault(kw, 0)
                        keywords[event_type][kw] += 1

        # save keywords (to be aggregated for use after serialization)
        if keywords_path:
            logger.info('Saving Positive Trainset Triggers to {} '.format(keywords_path))
            with codecs.open(keywords_path, 'w', encoding='utf8') as f:
                json.dump(keywords, f, sort_keys=True)
        self.positive_triggers = keywords  # set keywords for this process
        return examples  # pass through candidates as-is

    @staticmethod
    def _get_do_not_tag_as_none_dict(params):
        do_not_tag_as_none_dict = None
        if 'trigger.do_not_tag_as_none.file' in params:
            do_not_tag_as_none_dict = defaultdict(set)
            with codecs.open(params['trigger.do_not_tag_as_none.file'], 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    docid = tokens[0]
                    for w in tokens[1:]:
                        do_not_tag_as_none_dict[docid].add(w)
        return do_not_tag_as_none_dict

    def _read_event_keywords(self, filepath):
        """ Read event keywords from a JSON file
        :rtype: dict[str, set[str]]
        """
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        ret = dict()
        reverse_map = defaultdict(set)
        for data in datas:
            et_string = data['event_type']
            keywords = set(data['keywords'] + data['variants'] + data['hyponym_words'])
            ret[et_string] = keywords
            for kw in keywords:
                reverse_map[kw].add(et_string)
        return ret, reverse_map

    def _read_candidate_span_file(self, filepath):
        ret = defaultdict(set)

        filepaths = []
        with open(filepath, 'r') as f:
            for line in f:
                filepaths.append(line.strip())

        for fp in filepaths:
            with codecs.open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    docid = tokens[0]
                    offset = IntPair(int(tokens[1]), int(tokens[2]))
                    ret[docid].add(offset)
        return ret

    def _read_false_negatives(self, filepath):
        ret = dict()
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line[line.index(' ') + 1:]  # skip the first token/column
                docid_sentence_num = line[0:line.index(' ')]
                sentence_text = line[line.index(' ') + 1:]
                ret[docid_sentence_num] = sentence_text
        return ret

    def _apply_positive_training_trigger_filter_to_examples(self, examples):
        """
        REMOVE EXAMPLES NOT IN POSITIVE TRAINING TRIGGERS
        """
        all_NAs = 0
        kept_NAs = 0
        all_pos = 0
        kept_pos = 0
        new_examples = []
        logger.info("Filtering data subset examples to only include keywords found "
                    "in positive training triggers")
        for eg in examples:
            keyword = eg.get_keyword()
            for event_type in eg.event_type:
                if event_type == 'None':
                    all_NAs += 1
                else:
                    all_pos += 1
            if any(keyword in d for d in self.positive_triggers.values()):
                new_examples.append(eg)
                for event_type in eg.event_type:
                    if event_type == 'None':
                        kept_NAs += 1
                    else:
                        kept_pos += 1
        logger.info("KEPT {} of {} NA INSTANCES".format(kept_NAs, all_NAs))
        logger.info("KEPT {} of {} POSITIVE INSTANCES".format(kept_pos, all_pos))
        return new_examples

    def _downsample_na_examples(self, examples):
        """DOWNSAMPLE NA INSTANCES
        :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        """

        na_sample_rate = self.extractor_params.get("trigger.na_sample_rate", 3.0)
        number_of_positives = 0
        na_examples = []
        new_examples = []

        # 1. store positives and negatives separately
        for eg in examples:
            if eg.is_na():
                if self.extractor_params.get('trigger.use_safelist', False):
                    keyword = eg.get_keyword()
                    if not any(keyword in d for d in self.positive_triggers.values()):
                        continue
                na_examples.append(eg)
            else:
                new_examples.append(eg)
                number_of_positives += 1

        # 2. sample NAs
        number_of_nas = int(na_sample_rate * number_of_positives)
        random.shuffle(na_examples)
        new_examples.extend(na_examples[:number_of_nas])

        # 3. log statistics
        logger.info("NA:POSITIVE INSTANCES SAMPLING RATIO of {}:1.0 = {}:{}"
                    .format(na_sample_rate, number_of_nas, number_of_positives))
        logger.info("(Dropped {}/{} potential NA triggers)"
                    .format(len(na_examples) - number_of_nas, len(na_examples)))
        return new_examples

    def _remove_na_examples(self, examples):
        na_examples = 0
        positive_examples = []
        for eg in examples:
            if eg.is_na():
                na_examples += 1
            else:
                positive_examples.append(eg)
        logger.info("Removed all {} NA samples leaving {} samples."
                    .format(na_examples, len(positive_examples)))
        return positive_examples

    def _downsample_positive_examples_by_type_and_keyword(self, examples):
        """
        :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        """
        # should not be called if these parameters are missing
        max_per_type_and_keyword = self.extractor_params[
            "trigger.max_per_type_and_keyword"]
        examples_by_type_and_keyword = {}
        new_examples = set()
        na_examples = []
        logger.info("SAMPLING UP TO {} POSITIVE INSTANCES PER (TYPE, KEYWORD) "
                    "PAIR".format(max_per_type_and_keyword))

        # preparation loop: separate positives and negatives and keep key info
        for eg in examples:
            if eg.is_na():
                na_examples.append(eg)
            else:
                keyword = eg.get_keyword()
                types = eg.event_type

                for event_type in types:
                    k = (event_type, keyword)
                    examples_by_type_and_keyword.setdefault(k, list()).append(eg)

        # sampling loop
        all_positives = 0
        for k, v in examples_by_type_and_keyword.items():
            all_positives += len(v)
            random.shuffle(v)
            # max may not be reached due to multiple labels
            v = v[:max_per_type_and_keyword]
            new_examples.update(v)

        # log statistics and make output
        logger.info("SAMPLED {} of {} POSITIVE INSTANCES"
                    .format(len(new_examples), all_positives))
        new_examples = list(new_examples)
        new_examples.extend(na_examples)
        return new_examples

    def generate(self, docs):
        """
        Returns unfeaturized candidates.
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        self.statistics.clear()

        # 1. create candidates
        candidates = []
        for index, doc in enumerate(docs):
            for sent in doc.sentences:
                doc_sent_id = '{}_{}'.format(sent.docid, sent.index)
                if doc_sent_id not in self.sentence_ids_to_discard:
                    chunk = self._generate_sentence(sent)
                    """:type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""
                    candidates.extend(chunk)

            if (index % 20) == 0:
                logger.info('Generated examples from {} documents out of {}'
                            .format(str(index + 1), str(len(docs))))

        # 2. apply selected preprocessing filters
        filters = []
        if self.hyper_params.decode_mode:  # TODO will require fixing Hume templates

            # apply safelist-related filters
            if self.extractor_params.get('trigger.use_safelist', False):
                # simple filter based on existing safelist
                filters.append(self._apply_positive_training_trigger_filter_to_examples)

        else:  # training-time filters

            # Remove all NA candidates
            filters.append(self._remove_na_examples)

            # get keyword safelist
            if self.positive_triggers is None:
                filters.append(self._save_positive_trigger_keywords)
            # sample N positive examples per (type, keyword) pair
            filters.append(self._downsample_positive_examples_by_type_and_keyword)

            # # apply safelist-related filters
            # if self.extractor_params.get('trigger.use_safelist', False):
            #     # # simple filter based on safelist
            #     # filters.append(self._apply_positive_training_trigger_filter_to_examples)
            #     pass

            # # Downsample NA candidates, keeping a certain ratio of NA:positive
            # filters.append(self._downsample_na_examples)

        for filter_function in filters:
            logger.info("FILTERING EXAMPLES USING FUNCTION: {}"
                        .format(filter_function.__name__))
            candidates = filter_function(candidates)

        for k, v in self.statistics.items():
            logger.info('EventTriggerExampleGenerator stats, {}:{}'.format(k, v))

        return candidates

    def _generate_unigram_examples(self, sentence):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
        :type params: dict
        :type extractor_params: dict
        :type features: nlplingo.tasks.eventtrigger.feature.EventTriggerFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        ret = []
        for token_index, token in enumerate(sentence.tokens):

            event_types = EventTriggerFeatureGenerator.get_event_types_of_token(token, sentence)
            if event_types == {'None'}:
                self.statistics['POSTAG {} TN'.format(token.pos_tag)] += 1
                self.statistics['POSTAG-ALT {} TN'.format(token.pos_tag_alternate)] += 1
            else:
                self.statistics['POSTAG {} TP'.format(token.pos_tag)] += len(event_types)
                self.statistics['POSTAG-ALT {} TP'.format(token.pos_tag_alternate)] += len(event_types)
            event_types = set(
                t for t in event_types if self._accept_tokens_as_candidate(
                    [token], t, sentence.entity_mentions, sentence.docid))
            if len(event_types) == 0:
                continue

            self.statistics['number_candidate_trigger'] += 1
            self.statistics[token.pos_category()] += len(event_types - {'None'})
            self.statistics['number_positive_trigger'] += len(event_types - {'None'})

            # may break/need special handling!
            anchor_candidate = Anchor('dummy-id',
                                      IntPair(token.start_char_offset(),
                                              token.end_char_offset()),
                                      token.text, event_types)
            anchor_candidate.with_tokens([token])

            embedding_vector_size = self.extractor_params['embeddings']['vector_size']
            example = EventTriggerExample(
                anchor_candidate,
                sentence,
                self.event_domain,
                embedding_vector_size,
                event_types)

            ret.append(example)

        return ret

    def _use_sentence(self, sentence):
        if sentence.number_of_tokens() < 1:
            return False
        if sentence.number_of_tokens() > self.hyper_params.max_sentence_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return False

        if self.trigger_candidate_span is not None:
            start = sentence.tokens[0].start_char_offset()
            end = sentence.tokens[-1].end_char_offset()
            in_span = False
            for offset in self.trigger_candidate_span[sentence.docid]:
                if offset.first >= start and end >= offset.second:
                    in_span = True
                    break
            return in_span
        else:
            return True

    def _generate_sentence(self, sentence):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
        :rtype: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        """
        ret = []

        if not self._use_sentence(sentence):
            self.statistics['number_events_in_skipped_sentences'] += len(sentence.events)
            self.statistics['number_triggers_in_skipped_sentences'] += len([event.anchors for event in sentence.events])
            return ret

        self.statistics['number_event'] += len(sentence.events)
        # TODO I'm pretty sure this will always equal the number above, so if we have multi-anchor events this is no good.
        self.statistics['number_trigger'] += len([event.anchors for event in sentence.events])

        ret.extend(self._generate_unigram_examples(sentence))
        return ret

    def _accept_tokens_as_candidate(self, tokens, event_type, entity_mentions, docid):
        """Whether to reject a token as trigger candidate
        :type tokens: list[nlplingo.text.text_span.Token]
        :type entity_mentions: list[nlplingo.text.text_span.EntityMention]
        """
        if self.spans_to_generate_examples is not None:
            for offset in self.spans_to_generate_examples[docid]:
                if offset.first == tokens[0].start_char_offset() and offset.second == tokens[-1].end_char_offset():
                    return True
            if self.only_generate_annotated_spans:
                return False

        if tokens[-1].pos_category() not in trigger_pos_category:
            self.statistics['Reject trigger pos_category'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger pos_category: pos={} text={} {}'.format(tokens[-1].pos_tag, tokens[-1].text,
                                                                                   event_type))
                self.statistics['Reject trigger pos_category TP'] += 1
            else:
                return False
            #return False
        if "'" in tokens[-1].text:
            self.statistics['Reject trigger \''] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger \': text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger \' TP'] += 1
            return False
        if re.search('\d', tokens[-1].text):  # there's a digit
            self.statistics['Reject trigger digit'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger digit: text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger digit TP'] += 1
            return False
        if len(tokens[-1].text) < 2:
            self.statistics['Reject trigger len < 2'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger len < 2: text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger len < 2 TP'] += 1
            return False
        if tokens[-1].text.lower() in self.negative_trigger_words:
            self.statistics['Reject trigger negative-word'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger negative-word: text={} {}'.format(tokens[-1].text.lower(), event_type))
                self.statistics['Reject trigger negative-word TP'] += 1
            return False

        for em in entity_mentions:
            if em.head() is None:
                self.statistics['EM head is None'] += 1
                continue
            if self.extractor_params.get('trigger.allow_entity_mention_as_anchor', False):
                if em.head() == tokens[-1] and em.label != 'OTH':
                    self.statistics['Not rejected: trigger overlap-EM'] += 1
                    if event_type != 'None':
                        self.statistics['Not rejected: trigger overlap-EM TP'] += 1
            else:
                if em.head() == tokens[-1] and em.label != 'OTH':
                    self.statistics['Reject trigger overlap-EM'] += 1
                    if event_type != 'None':
                        safeprint(u'Reject trigger overlap-EM: docid={} text=({}) em-type={} {}'.format(docid, '_'.join(
                            token.text.lower() for token in tokens), em.label, event_type))
                        self.statistics['Reject trigger overlap-EM TP'] += 1
                    return False

        if self._candidate_is_illegal_none(event_type, docid, tokens):
            return False

        if self.trigger_candidate_span is not None:
            if docid not in self.trigger_candidate_span:
                self.statistics['Reject trigger trigger_candidate_span file does not contain docid'] += 1
                return False
            offsets = self.trigger_candidate_span[docid]
            for offset in offsets:
                if offset.first <= tokens[0].start_char_offset() and offset.second >= tokens[-1].end_char_offset():
                    return True
            self.statistics['Reject trigger trigger_candidate_span file does not contain trigger'] += 1
            return False

        return True

    def _candidate_is_illegal_none(self, event_type, docid, tokens):

        if self.do_not_tag_as_none_dict is not None:
            if event_type == 'None' and docid in self.do_not_tag_as_none_dict:
                if tokens[-1].text.lower() in self.do_not_tag_as_none_dict[docid]:
                    self.statistics['Reject trigger as None since keyword for event-type'] += 1
                    return True

        if self.restrict_none_examples_using_keywords and event_type == 'None':
            event_types = self.event_keyword_list.get_event_types_for_tokens(tokens)
            if len(event_types) > 0:
                self.statistics['Reject trigger as None since keyword for event-type'] += 1
                return True

        return False

