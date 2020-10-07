
import sys
import argparse
from collections import defaultdict

from nlplingo.tasks.sequence.example import SequenceExample
from serif.io.bpjson.abstract_events import Corpus

from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.text.text_span import LabeledTextSpan
from nlplingo.common.utils import IntPair
#anchor_lens = defaultdict(int)


class TextSpan(object):
    def __init__(self, text):
        self.text = text
        self.start_token_index = int
        self.end_token_index = int
        self.tokens = None

    @classmethod
    def from_full_info(cls, text, start_token_index, end_token_index, tokens):
        text_span = cls(text)
        text_span.start_token_index = int(start_token_index)
        text_span.end_token_index = int(end_token_index)
        text_span.tokens = tokens
        return text_span 

    @property
    def length(self):
        return len(self.text)


class TextMatcher(object):
    @staticmethod
    def _select_longest_spans(spans):
        """ From a list of TextSpan, return a list containing those spans with the longest texts

        :type spans: list[TextSpan]
        :rtype: list[TextSpan]
        """
        length_to_spans = defaultdict(list)
        max_len = 0

        for span in spans:
            length_to_spans[span.length].append(span)
            if span.length > max_len:
                max_len = span.length
        return length_to_spans[max_len]    

    @staticmethod
    def _select_shortest_spans(spans):
        """ From a list of TextSpan, return a list containing those spans with the shortest texts

        :type spans: list[TextSpan]
        :rtype: list[TextSpan]
        """
        length_to_spans = defaultdict(list)
        min_len = sys.maxsize

        for span in spans:
            length_to_spans[span.length].append(span)
            if span.length < min_len:
                min_len = span.length
        return length_to_spans[min_len]

    @staticmethod
    def _exact_match(tokens, target_tokens):
        """ Find exact sequence of 'target_tokens' within 'tokens'

        :type tokens: list[str]
        :type target_tokens: list[str]
        :rtype: (str, list[TextSpan])
        """
        ret = []
        target_len = len(target_tokens)

        for i in range(0, len(tokens)):
            if tokens[i : i+target_len] == target_tokens:
                text_span = TextSpan.from_full_info(' '.join(tokens[i : i+target_len]), i, i+target_len-1, tokens[i : i+target_len])
                ret.append(text_span)
        return 'exact_match', ret

    @staticmethod
    def _prefix_match(tokens, target_tokens):
        """ Find sequence within 'tokens', where each token within that sequence starts with each token within 'target_tokens'

        :type tokens: list[str]
        :type target_tokens: list[str]
        :rtype: (str, list[TextSpan])
        """
        ret = []
        target_len = len(target_tokens)

        for i in range(0, len(tokens) - target_len + 1):
            match = True
            for j in range(0, target_len):
                if not tokens[i+j].startswith(target_tokens[j]):
                    match = False
                    break
            if match:
                text_span = TextSpan.from_full_info(' '.join(tokens[i : i+target_len]), i, i+target_len-1, tokens[i : i+target_len])
                ret.append(text_span)
        return 'prefix_match', ret

    @staticmethod
    def _prefix_or_suffix_match(tokens, target_tokens):
        """ Find sequence within 'tokens', where each token within that sequence starts/ends with each token within 'target_tokens'

        :type tokens: list[str]
        :type target_tokens: list[str]
        :rtype: (str, list[TextSpan])
        """
        ret = []
        target_len = len(target_tokens)

        for i in range(0, len(tokens) - target_len + 1):
            match = True
            for j in range(0, target_len):
                if (not tokens[i+j].startswith(target_tokens[j])) and (not tokens[i+j].endswith(target_tokens[j])):
                    match = False
                    break
            if match:
                text_span = TextSpan.from_full_info(' '.join(tokens[i : i+target_len]), i, i+target_len-1, tokens[i : i+target_len])
                ret.append(text_span)
        return 'prefix_or_suffix_match', ret

    @staticmethod
    def _inclusion_match(tokens, target_tokens):
        """ Find sequence within 'tokens', where each token within that sequence includes each token within 'target_tokens'

        :type tokens: list[str]
        :type target_tokens: list[str]
        :rtype: (str, list[TextSpan])
        """
        ret = []
        target_len = len(target_tokens)

        for i in range(0, len(tokens) - target_len + 1):
            match = True
            for j in range(0, target_len):
                if not target_tokens[j] in tokens[i+j]:
                    match = False
                    break
            if match:
                text_span = TextSpan.from_full_info(' '.join(tokens[i : i+target_len]), i, i+target_len-1, tokens[i : i+target_len])
                ret.append(text_span)
        return 'inclusion_match', ret

    @staticmethod
    def locate_text_within_tokens(tokens, text):
        """
        :type tokens: list[str]
        :type text: str
        :rtype: (str, list[TextSpan])
        """
        # first, try an exact match
        match_strategy, ret = TextMatcher._exact_match(tokens, text.split())
        ret = TextMatcher._select_shortest_spans(ret)   # select the most precise matches
        if len(ret) > 0:
            return match_strategy, ret

        # try fuzzy match where we match using prefix
        match_strategy, ret = TextMatcher._prefix_match(tokens, text.split())
        ret = TextMatcher._select_shortest_spans(ret)
        if len(ret) > 0:
            return match_strategy, ret

        # try fuzzy match where we match using either prefix or suffix
        match_strategy, ret = TextMatcher._prefix_or_suffix_match(tokens, text.split())
        ret = TextMatcher._select_shortest_spans(ret)
        if len(ret) > 0:
            return match_strategy, ret

        # so long as target text is within string, e.g. 'excess' will match '"excess"'
        match_strategy, ret = TextMatcher._inclusion_match(tokens, text.split())
        ret = TextMatcher._select_shortest_spans(ret)
        if len(ret) > 0:
            return match_strategy, ret

        return 'unmatched', ret

# TODO this is deprecated and can be removed
class BIOGenerator(object):
    def __init__(self):
        self.num_sentences_without_annotation = 0
        self.max_seq_len = 0        # maximum of tokens we have seen in a sentence
        self.max_seq_text = None

    @staticmethod
    def _format_bio_outlines(doc_id, entry_id, text_tokens, bio_labels):
        """
        :type doc_id: str
        :type entry_id: str
        :type text_tokens: list[str]
        :type bio_labels: list[str]
        :rtype: str
        """
        id_header = '-DOCSTART- {} {}'.format(doc_id, entry_id)
        bio_annotation = '\n'.join('{} {}'.format(t, l) for t, l in zip(text_tokens, bio_labels))
        return '{}\n{}'.format(id_header, bio_annotation)

    def _has_annotation(self, bio_labels):
        """ Does this sentence have any annotation?
        :type bio_labels: list[str]
        """
        has_annotation = False
        for label in bio_labels:
            if label != 'O':
                has_annotation = True
                break
        return has_annotation

    def _capture_statistics(self, text_tokens, bio_labels):
        """
        :type text_tokens: list[str]
        :type bio_labels: list[str]
        """
        if len(text_tokens) > self.max_seq_len:
            self.max_seq_len = len(text_tokens)
            self.max_seq_text = ' '.join(text_tokens)
           
        # does this sentence have any annotation?
        if not self._has_annotation(bio_labels):
            self.num_sentences_without_annotation += 1

    def _print_statistics(self):
        print('max_seq_len={}'.format(str(self.max_seq_len)))
        #print('max_seq_text={}'.format(str(self.max_seq_text)))
        print('number of sentences without annotation = %d' % (self.num_sentences_without_annotation))

    def id_to_lingo_sentences(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :rtype: dict[str, nlplingo.text.text_span.Sentence]
        """
        ret = dict()
        for doc in docs:
            assert len(doc.sentences) == 1
            ret[doc.docid] = doc.sentences[0]

        return ret

    def generate_trigger_bio(self, bp_json_file, keep_unannotated_sentences, tokenization_type, nlplingo_docs):
        """
        :type bp_json_file: str
        :type keep_unannotated_sentences: bool
        :type tokenization_type: str
        :type nlplingo_docs: list[nlplingo.text.text_theory.Document]
        """
        bp_corpus = Corpus(bp_json_file)

        id_to_lingo_sentences = self.id_to_lingo_sentences(nlplingo_docs)

        self.num_sentences_without_annotation = 0
        self.max_seq_len = 0
        outlines = []
        for _, doc in bp_corpus.docs.items():
            for sentence in doc.sentences:
                if sentence.entry_id not in id_to_lingo_sentences:
                    continue
                #assert sentence.entry_id in id_to_lingo_sentences
                lingo_sentence = id_to_lingo_sentences[sentence.entry_id]

                if tokenization_type == 'SPACE':
                    text_tokens = lingo_sentence.text.split()
                    #text_tokens = sentence.text.split()
                elif tokenization_type == 'SERIF':
                    text_tokens = [token.text for token in lingo_sentence.tokens]

                bio_labels = ['O'] * len(text_tokens)

                match_strategies = set()
                for _, abstract_event in sentence.abstract_events.items():
                    for anchor_span in abstract_event.anchors.spans:
                        anchor_text = anchor_span.string
                        event_type = '{}.{}'.format(abstract_event.helpful_harmful, abstract_event.material_verbal)

                        match_strategy, anchor_matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, anchor_text)
                        """:type (str, list[TextSpan])"""

                        match_strategies.add(match_strategy)

                        for matched_span in anchor_matched_spans:
                            bio_labels[matched_span.start_token_index] = 'B-{}'.format(event_type)
                            for i in range(matched_span.start_token_index+1, matched_span.end_token_index+1):
                                bio_labels[i] = 'I-{}'.format(event_type)

                if keep_unannotated_sentences:
                    outlines.append(BIOGenerator._format_bio_outlines(doc.doc_id, sentence.entry_id, text_tokens, bio_labels))
                else:
                    if self._has_annotation(bio_labels):
                        outlines.append(BIOGenerator._format_bio_outlines(doc.doc_id, sentence.entry_id, text_tokens, bio_labels))
                self._capture_statistics(text_tokens, bio_labels)

        self._print_statistics()

        return outlines
        # with open(args.output_file, 'w', encoding='utf-8') as o:
        #     o.write('\n\n'.join(outlines))
        #     o.write('\n')


    def _annotate_bio_for_argument(self, arguments, text_tokens, bio_labels, role):
        texts = set()
        match_strategies = set()

        for arg in arguments:
            for span in arg.spans:
                texts.add(span.string)
                match_strategy, m_spans = TextMatcher.locate_text_within_tokens(text_tokens, span.string)
                match_strategies.add(match_strategy)
                for m_span in m_spans:		# for each matching span
                    bio_labels[m_span.start_token_index] = 'B-{}'.format(role)
                    for i in range(m_span.start_token_index+1, m_span.end_token_index+1):
                        bio_labels[i] = 'I-{}'.format(role)

        return match_strategies, texts

    def _extend_with_anchor_annotation(self, text_tokens, bio_labels, anchor_start, anchor_end):
        """
        :type text_tokens: list[str]
        :type bio_labels: list[str]
        :type anchor_start: int
        :type anchor_end: int
        """
        eg_texts = []
        eg_labels = []

        eg_texts.extend(text_tokens[0:anchor_start])
        eg_labels.extend(bio_labels[0:anchor_start])
        eg_texts.append('$$$')
        eg_labels.append('O')
        eg_texts.extend(text_tokens[anchor_start:anchor_end+1])
        eg_labels.extend(['O'] * (anchor_end - anchor_start+1))
        eg_texts.append('$$$')
        eg_labels.append('O') 
        eg_texts.extend(text_tokens[anchor_end+1:])
        eg_labels.extend(bio_labels[anchor_end+1:])

        return (eg_texts, eg_labels)

    def generate_argument_bio(self, bp_json_file, keep_unannotated_sentences, tokenization_type, nlplingo_docs):
        """
        :type bp_json_file: str
        :type keep_unannotated_sentences: bool
        :type tokenization_type: str
        :type nlplingo_docs: list[nlplingo.text.text_theory.Document]
        """
        bp_corpus = Corpus(bp_json_file)

        id_to_lingo_sentences = self.id_to_lingo_sentences(nlplingo_docs)

        self.num_sentences_without_annotation = 0
        self.max_seq_len = 0
        outlines = []
        for _, doc in bp_corpus.docs.items():
            for sentence in doc.sentences:
                assert sentence.entry_id in id_to_lingo_sentences

                lingo_sentence = id_to_lingo_sentences[sentence.entry_id]

                for _, abstract_event in sentence.abstract_events.items():
                    if tokenization_type == 'SPACE':
                        text_tokens = lingo_sentence.text.split()
                        #text_tokens = sentence.text.split()
                    elif tokenization_type == 'SERIF':
                        text_tokens = [token.text for token in lingo_sentence.tokens]

                    bio_labels = ['O'] * len(text_tokens)

                    self._annotate_bio_for_argument(abstract_event.patients, text_tokens, bio_labels, 'PATIENT')
                    self._annotate_bio_for_argument(abstract_event.agents, text_tokens, bio_labels, 'AGENT')

                    for anchor_span in abstract_event.anchors.spans:
                        anchor_text = anchor_span.string
                        #event_type = '{}.{}'.format(abstract_event.helpful_harmful, abstract_event.material_verbal)

                        match_strategy, anchor_matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, anchor_text)

                        # if len(anchor_matched_spans) > 1, then it is ambiguous which anchor, that the arguments should attach to
                        for anchor_matched_span in anchor_matched_spans:
                            anchor_start = anchor_matched_span.start_token_index
                            anchor_end = anchor_matched_span.end_token_index

                            eg_texts, eg_labels = self._extend_with_anchor_annotation(text_tokens, bio_labels, anchor_start, anchor_end)
                            if keep_unannotated_sentences:
                                outlines.append(BIOGenerator._format_bio_outlines(doc.doc_id, sentence.entry_id, eg_texts, eg_labels))
                            else:
                                if self._has_annotation(eg_labels):
                                    outlines.append(BIOGenerator._format_bio_outlines(doc.doc_id, sentence.entry_id, eg_texts, eg_labels))
                            self._capture_statistics(eg_texts, eg_labels)

        self._print_statistics()

        return outlines
        # with open(args.output_file, 'w', encoding='utf-8') as o:
        #     o.write('\n\n'.join(outlines))
        #     o.write('\n')


def align_tokens(source_toks, target_toks):
    # We will assume that target_tokens is a tokenized version of source_tokens,
    # e.g. source_tokens from raw sentence, target_tokens from SERIF tokenization of the raw sentence
    #:type source_tokens: list[str]
    #:type target_tokens: list[str]

    unicode = '\u202b'

    source_tokens = []
    for token in source_toks:
        token = token.replace(unicode, '')
        source_tokens.append(token)

    target_tokens = []
    for token in target_toks:
        token = token.replace(unicode, '')
        target_tokens.append(token)

    # first, check that they have the same texts
    source_len = 0
    for token in source_tokens:
        source_len += len(token)
    target_len = 0
    for token in target_tokens:
        target_len += len(token)

    if source_len != target_len:
        print('WARNING! source_len != target_len, source_len=', source_len, ', target_len=', target_len)
        print('SOURCE:', ' '.join(source_tokens))
        print('TARGET:', ' '.join(target_tokens))

    # assert source_len == target_len

    alignments = []  # list of tuples = (start_token_index, end_token_index), len(alignments) == len(source_tokens)
    start_index = 0
    source_len_thus_far = 0
    target_len_thus_far = 0
    for source_index, source_token in enumerate(source_tokens):
        source_len_thus_far += len(source_token)

        end_index = start_index
        while end_index < len(target_tokens) and (
                target_len_thus_far + len(target_tokens[end_index]) <= source_len_thus_far):
            target_len_thus_far += len(target_tokens[end_index])
            end_index += 1

        alignments.append((start_index, end_index - 1))
        start_index = end_index

    # print('SOURCE:', ' '.join('{}:{}-{}'.format(token, str(alignments[i][0]), str(alignments[i][1])) for i, token in enumerate(source_tokens)))
    # print('TARGET:', ' '.join('{}:{}'.format(str(i), token) for i, token in enumerate(target_tokens)))

    return alignments

def get_ner_annotations_from_docs(docs, target_types):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    :type target_types = set(str)       ; only target entity mentions of these types, e.g. PER, ORG, etc.
    :rtype: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]
    """
    span_annotations = dict()
    for doc in docs:
        doc_annotations = []

        for sentence in doc.sentences:
            token_alignments = align_tokens(sentence.text.split(), [token.text for token in sentence.tokens])
            # each token from 'sentence.text.split()' is aligned to token(s) from sentence.tokens
            assert len(token_alignments) == len(sentence.text.split())

            sentence_annotations = []
            for em in sentence.entity_mentions:

                # TODO the following code should probably be factorized into a generic method
                start_token_index = None    # what tokens in sentence.text.split() cover the entity mention?
                end_token_index = None
                for i, (start_index, end_index) in enumerate(token_alignments):
                    if start_index <= em.tokens[0].index_in_sentence <= end_index:
                        start_token_index = i
                    if start_index <= em.tokens[-1].index_in_sentence <= end_index:
                        end_token_index = i
                #assert start_token_index is not None
                #assert end_token_index is not None

                if start_token_index is None or end_token_index is None:
                    print('WARNING! skipping entity-mention because start_token_index=', start_token_index, ', end_token_index=', end_token_index)
                    continue

                if em.label in target_types:
                    span = LabeledTextSpan(IntPair(None, None), em.text, em.label)
                    span.start_token_index = start_token_index
                    span.end_token_index = end_token_index
                    sentence_annotations.append(span)

            doc_annotations.append(sentence_annotations)

        span_annotations[doc.docid] = doc_annotations
        assert len(doc_annotations) == len(doc.sentences)

    return span_annotations

class SequenceExampleGenerator(ExampleGenerator):
    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class to generate candidate Datapoint objects for the sequence
        task.  This class differs from the Generator classes for other tasks.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        :type event_domain: nlplingo.tasks.event_domain.EventDomain
        """
        super(SequenceExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)
        self.statistics = defaultdict(int)
        self.logs = defaultdict(list)

    def generate(self, docs):
        if not self.decode_mode:
            annotations = get_ner_annotations_from_docs(docs, set(k for k in self.event_domain.entity_types.keys() if k != 'None'))

            ret = []

            for doc in docs:
                doc_annotation = annotations.get(doc.docid, [])
                assert len(doc_annotation) == len(
                    doc.sentences)  # each item in doc_annotation corresponds to annotation on each sentence

                for sentence_index, sentence in enumerate(doc.sentences):
                    sentence_annotation = doc_annotation[sentence_index]
                    """:type: list[nlplingo.text.text_span.LabeledTextSpan]"""

                    text_tokens = [token.text for token in sentence.tokens]  # SERIF tokens
                    # text_tokens = sentence.text.split()                         # SPACE tokens
                    bio_labels = ['O'] * len(text_tokens)

                    for annotation in sentence_annotation:
                        self.statistics['label={}'.format(annotation.label)] += 1

                        if annotation.start_token_index is not None and annotation.end_token_index is not None:
                            span_length = annotation.end_token_index - annotation.start_token_index + 1
                            self.statistics['span_token_length={}'.format(str(span_length))] += 1
                            if span_length >= 3:  # will be interesting to log these
                                self.logs['anchor_span_length={}'.format(str(span_length))].append(
                                    '{} [{}]'.format(doc.docid, ' '.join(
                                        text_tokens[annotation.start_token_index:annotation.end_token_index + 1])))

                            assert annotation.start_token_index < len(bio_labels)
                            assert annotation.end_token_index < len(bio_labels)
                            bio_labels[annotation.start_token_index] = 'B-{}'.format(annotation.label)
                            for i in range(annotation.start_token_index + 1, annotation.end_token_index + 1):
                                bio_labels[i] = 'I-{}'.format(annotation.label)
                        else:
                            _, matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, annotation.text)
                            """:type (str, list[TextSpan])"""

                            for matched_span in matched_spans:
                                self.statistics['span_token_length={}'.format(
                                    str(matched_span.end_token_index - matched_span.start_token_index + 1))] += 1

                                bio_labels[matched_span.start_token_index] = 'B-{}'.format(annotation.label)
                                for i in range(matched_span.start_token_index + 1, matched_span.end_token_index + 1):
                                    bio_labels[i] = 'I-{}'.format(annotation.label)

                    example = SequenceExample(sentence.text, text_tokens, bio_labels, sentence, self.event_domain)
                    example.docid = doc.docid
                    example.sentence_index = sentence_index
                    # feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                    ret.append(example)

                    self.statistics['num# example'] += 1

            return ret
        else:
            ret = []

            for doc in docs:
                for i, sentence in enumerate(doc.sentences):
                    text_tokens = [token.text for token in sentence.tokens]    # SERIF tokenization
                    #text_tokens = sentence.text.split()                         # SPACE tokenization
                    bio_labels = ['O'] * len(text_tokens)

                    example = SequenceExample(sentence.text, text_tokens, bio_labels, sentence, self.event_domain)
                    example.docid = doc.docid
                    example.sentence_index = i

                    #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                    ret.append(example)

            return ret
    # +1
    def generate_spans_for_training(self, docs, annotations, use_only_begin_tag=False):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type annotations: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]
        :type feature_generator: nlplingo.tasks.sequence.feature.SequenceFeatureGenerator

        :rtype: list[nlplingo.tasks.sequence.example.SequenceExample]

        param annotations:
        * provide the annotations to generate for.
        * maps from docid to a list (representing a sentence) of list (TextSpan to annotate in that sentence).
        """
        ret = []

        for doc in docs:
            doc_annotation = annotations.get(doc.docid, [])
            assert len(doc_annotation) == len(doc.sentences)    # each item in doc_annotation corresponds to annotation on each sentence

            for sentence_index, sentence in enumerate(doc.sentences):
                sentence_annotation = doc_annotation[sentence_index]
                """:type: list[nlplingo.text.text_span.LabeledTextSpan]"""

                text_tokens = [token.text for token in sentence.tokens]    # SERIF tokens
                #text_tokens = sentence.text.split()                         # SPACE tokens
                bio_labels = ['O'] * len(text_tokens)

                for annotation in sentence_annotation:
                    self.statistics['label={}'.format(annotation.label)] += 1

                    if annotation.start_token_index is not None and annotation.end_token_index is not None:
                        span_length = annotation.end_token_index - annotation.start_token_index + 1
                        self.statistics['span_token_length={}'.format(str(span_length))] += 1
                        if span_length >= 3:  # will be interesting to log these
                            self.logs['anchor_span_length={}'.format(str(span_length))].append('{} [{}]'.format(doc.docid, ' '.join(text_tokens[annotation.start_token_index:annotation.end_token_index + 1])))

                        assert annotation.start_token_index < len(bio_labels)
                        assert annotation.end_token_index < len(bio_labels)

                        if use_only_begin_tag:
                            bio_labels[annotation.start_token_index] = annotation.label
                        else:
                            bio_labels[annotation.start_token_index] = 'B-{}'.format(annotation.label)
                            for i in range(annotation.start_token_index + 1, annotation.end_token_index + 1):
                                bio_labels[i] = 'I-{}'.format(annotation.label)
                    else:
                        _, matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, annotation.text)
                        """:type (str, list[TextSpan])"""

                        for matched_span in matched_spans:
                            self.statistics['span_token_length={}'.format(str(matched_span.end_token_index - matched_span.start_token_index + 1))] += 1

                            if use_only_begin_tag:
                                bio_labels[matched_span.start_token_index] = annotation.label
                            else:
                                bio_labels[matched_span.start_token_index] = 'B-{}'.format(annotation.label)
                                for i in range(matched_span.start_token_index + 1, matched_span.end_token_index + 1):
                                    bio_labels[i] = 'I-{}'.format(annotation.label)

                example = SequenceExample(sentence.text, text_tokens, bio_labels, sentence, self.event_domain)
                example.docid = doc.docid
                example.sentence_index = sentence_index
                #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                ret.append(example)

                self.statistics['num# example'] += 1

        return ret

    # +1
    def generate_frames_for_training(self, docs, annotations):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type annotations: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]
        :type feature_generator: nlplingo.tasks.sequence.feature.SequenceFeatureGenerator

        :rtype: list[nlplingo.tasks.sequence.example.SequenceExample]

        param annotations:
        * provide the annotations to generate for.
        * maps from docid to a list (representing a sentence) of list (LabeledTextFrame to annotate in that sentence).
        """
        ret = []

        for doc in docs:
            doc_annotation = annotations.get(doc.docid, [])
            assert len(doc_annotation) == len(doc.sentences)  # each item in doc_annotation corresponds to annotation on each sentence

            for sentence_index, sentence in enumerate(doc.sentences):
                sentence_annotation = doc_annotation[sentence_index]
                """:type: list[nlplingo.text.text_span.LabeledTextFrame]"""

                text_tokens = [token.text for token in sentence.tokens]    # SERIF tokens
                #text_tokens = sentence.text.split()                         # SPACE tokens

                for annotation in sentence_annotation:      # for each LabeledTextFrame
                    bio_labels = ['O'] * len(text_tokens)

                    # first annotate the bio_labels with argument span labels
                    for span in annotation.argument_spans:
                        self.statistics['label={}'.format(span.label)] += 1

                        if span.start_token_index is not None and span.end_token_index is not None:
                            span_length = span.end_token_index - span.start_token_index + 1
                            self.statistics['argument_span_token_length={}'.format(str(span_length))] += 1
                            if span_length >= 10:       # will be interesting to log these
                                self.logs['argument_span_length={}'.format(str(span_length))].append('{} [{}]'.format(doc.docid, ' '.join(text_tokens[span.start_token_index:span.end_token_index+1])))

                            bio_labels[span.start_token_index] = 'B-{}'.format(span.label)
                            for i in range(span.start_token_index + 1, span.end_token_index + 1):
                                bio_labels[i] = 'I-{}'.format(span.label)
                        else:
                            _, matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, span.text)
                            """:type (str, list[TextSpan])"""

                            for matched_span in matched_spans:
                                self.statistics['argument_span_token_length={}'.format(str(matched_span.end_token_index - matched_span.start_token_index + 1))] += 1

                                bio_labels[matched_span.start_token_index] = 'B-{}'.format(span.label)
                                for i in range(matched_span.start_token_index + 1, matched_span.end_token_index + 1):
                                    bio_labels[i] = 'I-{}'.format(span.label)

                    # now insert into bio_labels the anchor labels
                    for span in annotation.anchor_spans:
                        self.statistics['label={}'.format(span.label)] += 1

                        if span.start_token_index is not None and span.end_token_index is not None:
                            span_length = span.end_token_index - span.start_token_index + 1
                            self.statistics['anchor_span_token_length={}'.format(str(span_length))] += 1
                            if span_length >= 3:       # will be interesting to log these
                                self.logs['anchor_span_length={}'.format(str(span_length))].append('{} [{}]'.format(doc.docid, ' '.join(text_tokens[span.start_token_index:span.end_token_index+1])))

                            eg_texts, eg_labels = self._extend_with_anchor_annotation(text_tokens, bio_labels, span.start_token_index, span.end_token_index)
                            example = SequenceExample(sentence.text, eg_texts, eg_labels, sentence, self.event_domain)
                            example.docid = doc.docid
                            example.sentence_index = sentence_index
                            #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                            ret.append(example)
                            self.statistics['num# example'] += 1
                        else:
                            _, matched_spans = TextMatcher.locate_text_within_tokens(text_tokens, span.text)
                            """:type (str, list[TextSpan])"""

                            # if len(anchor_matched_spans) > 1, then it is ambiguous which anchor that the arguments should attach to
                            for matched_span in matched_spans:
                                self.statistics['anchor_span_token_length={}'.format(str(matched_span.end_token_index - matched_span.start_token_index + 1))] += 1

                                eg_texts, eg_labels = self._extend_with_anchor_annotation(text_tokens, bio_labels, matched_span.start_token_index, matched_span.end_token_index)
                                example = SequenceExample(sentence.text, eg_texts, eg_labels, sentence, self.event_domain)
                                example.docid = doc.docid
                                example.sentence_index = sentence_index
                                #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                                ret.append(example)
                                self.statistics['num# example'] += 1

        return ret

    # +1
    def _extend_with_anchor_annotation(self, text_tokens, bio_labels, anchor_start, anchor_end):
        """
        :type text_tokens: list[str]
        :type bio_labels: list[str]
        :type anchor_start: int
        :type anchor_end: int
        """
        eg_texts = []
        eg_labels = []

        eg_texts.extend(text_tokens[0:anchor_start])
        eg_labels.extend(bio_labels[0:anchor_start])
        eg_texts.append('$$$')
        eg_labels.append('O')
        eg_texts.extend(text_tokens[anchor_start:anchor_end+1])
        eg_labels.extend(['O'] * (anchor_end - anchor_start+1))
        eg_texts.append('$$$')
        eg_labels.append('O')
        eg_texts.extend(text_tokens[anchor_end+1:])
        eg_labels.extend(bio_labels[anchor_end+1:])

        return (eg_texts, eg_labels)

    # +1
    def generate_spans_for_decoding(self, docs):
        """ Generates an unlabeled SequenceExample for each sentence
        :type docs: list[nlplingo.text.text_theory.Document]
        :rtype: list[nlplingo.tasks.sequence.example.SequenceExample]
        """
        ret = []

        for doc in docs:
            for i, sentence in enumerate(doc.sentences):
                text_tokens = [token.text for token in sentence.tokens]    # SERIF tokenization
                #text_tokens = sentence.text.split()                         # SPACE tokenization
                bio_labels = ['O'] * len(text_tokens)

                example = SequenceExample(sentence.text, text_tokens, bio_labels, sentence, self.event_domain)
                example.docid = doc.docid
                example.sentence_index = i

                #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                ret.append(example)

        return ret

    # +1
    def generate_frames_for_decoding(self, docs, annotations):
        """ For each annotated anchor, generate a SequenceExample with anchor span surrounded by '$$$'

        :type docs: list[nlplingo.text.text_theory.Document]
        :type annotations: dict[str, list[list[LabeledTextSpan]]]
        :rtype: list[nlplingo.tasks.sequence.example.SequenceExample], list[str]

        annotations:
        * {docid} to a list, where len(list) == number of sentences in the doc
        * Each element in list is a list of LabeledTextSpan in that sentence
        """
        ret = []
        anchor_labels = []

        for doc in docs:
            assert doc.docid in annotations
            doc_annotation = annotations.get(doc.docid)
            assert len(doc_annotation) == len(doc.sentences)

            for i, sentence in enumerate(doc.sentences):
                sentence_annotation = doc_annotation[i]
                """:type: list[nlplingo.text.text_span.LabeledTextSpan]"""

                text_tokens = [token.text for token in sentence.tokens]    # SERIF tokenization
                #text_tokens = sentence.text.split()                         # SPACE tokenization
                bio_labels = ['O'] * len(text_tokens)

                for span in sentence_annotation:      # for each LabeledTextSpan
                    texts, labels = self._extend_with_anchor_annotation(text_tokens, bio_labels, span.start_token_index, span.end_token_index)
                    assert len(texts) == len(labels)
                    example = SequenceExample(sentence.text, texts, labels, sentence, self.event_domain)
                    example.docid = doc.docid
                    example.sentence_index = i

                    #feature_generator.generate_example(example, self.event_domain.sequence_types, tokenizer)
                    ret.append(example)
                    anchor_labels.append(span.label)

        return ret, anchor_labels

    def print_statistics(self):
        print('#### SequenceExampleGenerator statistics')
        for k in sorted(self.statistics):
            print(k, self.statistics[k])

        print('#### SequenceExampleGenerator logs')
        for k in sorted(self.logs):
            print(k)
            for v in self.logs[k]:
                print('*', v)