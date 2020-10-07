from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

import re
import logging

from nlplingo.common.utils import Struct
from nlplingo.common.utils import IntPair
from nlplingo.common.data_types import int_type, float_type

import numpy as np


logger = logging.getLogger(__name__)

span_types = Struct(TEXT='TEXT', SENTENCE='SENTENCE', TOKEN='TOKEN', ENTITY_MENTION='ENTITY_MENTION',
                    EVENTSPAN='EVENTSPAN', ANCHOR='ANCHOR', EVENT_ARGUMENT='EVENT_ARGUMENT')

punctuations = {'.', '?', '!', ',', ';', ':', '-', '(', ')', '{', '}', '[', ']', '<', '>', '"', "'", "`", '/', '~', '@',
                '#', '^', '&', '*', '+', '=', '_', '\\', '|'}

"""Classes here:
Span (this is abstract)

EntityMention(Span)
Anchor(Span)
EventSpan(Span)
EventArgument(Span)

Token(Span)
Sentence(Span)
"""

class IntegerPair(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def has_same_boundary(self, other):
        """
        :type other: IntegerPair
        """
        if self.start == other.start and self.end == other.end:
            return True
        else:
            return False

    def contains(self, other):
        """
        :type other: IntegerPair
        """
        if self.start <= other.start and other.end <= self.end:
            return True
        else:
            return False

    def has_overlapping_boundary(self, other):
        """
        :type other: IntegerPair
        """
        if (self.start <= other.start and other.start <= self.end) or \
                (other.start <= self.start and self.start <= other.end) or \
                self.has_same_boundary(other) or self.contains(other) or other.contains(self):
            return True
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    def to_string(self):
        return '({},{})'.format(self.start, self.end)


class Span(object):
    """An abstract class. A Span can be any sequence of text
    :int_pair: representing the start and end character offsets of this span
    :text: the text of this span
    """

    __metaclass__ = ABCMeta

    def __init__(self, int_pair, text):
        self.int_pair = IntPair(int_pair.first, int_pair.second)
        """:type: IntPair"""
        self.text = text

    def start_char_offset(self):
        return self.int_pair.first

    def end_char_offset(self):
        return self.int_pair.second

    @abstractmethod
    def span_type(self):
        """Return a string representing the type of span this is."""
        pass

    def contains(self, other_span):
        """
        :type other_span: nlplingo.text.text_span.Span
        :rtype: boolean
        """
        return self.start_char_offset() <= other_span.start_char_offset() and other_span.end_char_offset() <= self.end_char_offset()


class TextSpan(Span):
    """A simple span of texts"""
    def __init__(self, int_pair, text):
        Span.__init__(self, int_pair, text)
        self.start_token_index = None
        self.end_token_index = None

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def span_type(self):
        return span_types.TEXT


class LabeledTextSpan(TextSpan):
    def __init__(self, int_pair, text, label):
        super(LabeledTextSpan, self).__init__(int_pair, text)
        self.label = label

    def to_string(self):
        return '{}[{}]'.format(self.label, self.text)

class LabeledTextFrame(object):
    """A text frame is a grouping of several related pieces of text spans.
    There are anchor spans, and argument spans.
    An anchor span should not be interpreted to just mean event anchors.
    For instance, in sequence labeling, you might have a 2-stage pipeline process where you:
    * For event extraction:
       (i)  First predict triggers.
       (ii) The predicted triggers then serve as anchor spans to predict all arguments associated with those triggers
    * For relation extraction:
       (i)  First predict a core entity of interest.
       (ii) The core entity then serve to "anchor" subsequent predictions of other associated entities (spans).

    In a LabeledTextFrame, anchor_spans is a list. This allows the annotation data to specify multiple text spans as anchors.
    However, in decoding, we always have just a single anchor_span. If there are multiple anchor_spans,
    then it is ambiguous which anchor_span will a given argument_span be associated with.
    """
    def __init__(self, anchor_spans, argument_spans):
        """
        :type anchor_spans: list[nlplingo.text.text_span.LabeledTextSpan]
        :type argument_spans: list[nlplingo.text.text_span.LabeledTextSpan]
        """
        self.anchor_spans = anchor_spans
        self.argument_spans = argument_spans


class EntityMention(Span):
    """A consecutive span representing an entity mention.
    label: the NER type, e.g. Person, Organization, GPE
    """

    time_words = ['time', 'a.m.', 'am', 'p.m.', 'pm', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year',
                  'years', 'morning', 'afternoon', 'evening', 'night', 'anniversary', 'second', 'seconds', 'minute',
                  'minutes', 'hour', 'hours', 'decade', 'decades', 'january', 'february', 'march', 'april', 'may',
                  'june', 'july', 'august', 'september', 'october', 'november', 'december', 'today', 'yesterday',
                  'tomorrow', 'past', 'future', 'present', 'jan', 'jan.', 'feb', 'feb.', 'mar', 'mar.', 'apr', 'apr.',
                  'jun', 'jun.', 'jul', 'jul.', 'aug', 'aug.', 'sept', 'sept.', 'oct', 'oct.', 'nov', 'nov.', 'dec',
                  'dec.']

    def __init__(self, id, int_pair, text, label, entity=None, mention_type=None):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""
        self.head_token = None
        """:type: nlplingo.text.text_span.Token"""
        self.entity = entity
        self.mention_type = mention_type

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def coarse_label(self):
        if '.' in self.label:
            type = re.match(r'^(.*?)\.', self.label).group(1)   # get coarse grained NE type
        else:
            type = self.label
        return type

    @staticmethod
    def find_first_word_before(tokens, markers):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type markers: set[str]
        """
        for i, token in enumerate(tokens):
            if token.text.lower() in markers and i > 0:
                return tokens[i-1]
        return None

    def head(self):
        """
        Strategy for multi-word entity mentions.
        - Crime :
            (i) if there is a verb, use it as headword
            (ii) if there is 'of' or 'to', use the word before as the head
            (iii) else, use last word as head
        - Job-Title :
            (i) if there is a 'of' or 'at', use the word before as the head
            (ii) else, use last word as head
        - Numeric : use last word as head
        - Sentence : use last word as head
        - Time : look for the words in time_words (in order) and use it as the head if found. Else, use last word.
        - All other NE types:
            (i) remove any token consisting of just numbers and periods
            (ii) use last word as head

        Returns:
            nlplingo.text.text_span.Token
        """
        if self.head_token is not None:
            return self.head_token

        if self.tokens is None: # some entity mentions are not backed by tokens
            return None

        if len(self.tokens) == 1:
            return self.tokens[0]

        type = self.coarse_label()

        if type == 'Crime':
            for token in self.tokens:
                if token.pos_category() == u'VERB':
                    return token
            t = self.find_first_word_before(self.tokens, set(['of', 'to']))
            if t is not None:
                return t
            else:
                return self.tokens[-1]
        elif type == 'Job-Title':
            t = self.find_first_word_before(self.tokens, set(['of', 'at']))
            if t is not None:
                return t
            else:
                return self.tokens[-1]
        elif type == 'Numeric' or type == 'Sentence':
            return self.tokens[-1]
        elif type == 'Time':
            for w in self.time_words:
                for token in self.tokens:
                    if token.text.lower() == w:
                        return token
            return self.tokens[-1]
        else:
            for i, token in enumerate(self.tokens):
                if token.text.lower() == 'of' and i > 0:
                    return self.tokens[i-1]
                
            toks = []
            for token in self.tokens:
                if re.search(r'[a-zA-Z]', token.text) is not None:
                    toks.append(token)
            if len(toks) > 0:
                return toks[-1]
            else:
                return self.tokens[-1]

    def is_proper_noun(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'NNP' or head_pos_tag == 'NNPS'

    def is_common_noun(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'NN' or head_pos_tag == 'NNS'

    def is_adjective(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'JJ'

    def is_noun(self):
        return self.is_proper_noun() or self.is_common_noun()

    def head_pos_category(self):
        if self.is_proper_noun():
            return 'P'
        elif self.is_common_noun():
            return 'N'
        else:
            return '?'

    #def is_proper_or_common(self):
    #    if len(self.tokens) == 1:
    #        pos_tag = self.tokens[0].spacy_token.tag_
    #        # we do the following because some NNP (e.g. Chinese, Czech) are mis-tagged as JJ
    #        if pos_tag == 'PRP' or pos_tag == 'PRP$' or pos_tag == 'WP' or pos_tag == 'CD' or pos_tag == 'WDT':
    #           return False
    #    return True

    def span_type(self):
        return span_types.ENTITY_MENTION

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))

    def to_json(self):
        d = dict()
        d['id'] = self.id
        d['start'] = self.start_char_offset()
        d['end'] = self.end_char_offset()
        d['text'] = self.text
        d['label'] = self.label
        d['entity_id'] = self.entity.id
        if self.tokens:
            d['tokens'] = [t.to_json() for t in self.tokens]
        return d


class Anchor(Span):
    """A consecutive span representing an anchor
    label: the event type represented by the anchor, e.g. Conflict.Attack, CyberAttack, Vulnerability
    """

    def __init__(self, id, int_pair, text, label):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def head(self):
        """If the anchor is just a single token, we return the single token.
        If the anchor is multi-words, we heuristically determine a single token as the head

        Returns:
            nlplingo.text.text_span.Token
        """
        if len(self.tokens) == 1:
            return self.tokens[0]
        else:
            if self.tokens[0].pos_category() == 'VERB':
                return self.tokens[0]
            elif self.tokens[-1].pos_category() == 'NOUN' or self.tokens[-1].pos_category() == 'PROPN':
                return self.tokens[-1]
            else:
                for token in self.tokens:
                    if token.pos_category() == 'VERB' or token.pos_category() == 'NOUN' or token.pos_category() == 'PROPN':
                        return token
                return self.tokens[-1]

    def span_type(self):
        return span_types.ANCHOR

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))

    def to_json(self):
        d = dict()
        d['start'] = self.start_char_offset()
        d['end'] = self.end_char_offset()
        d['text'] = self.text
        d['label'] = self.label
        return d


class EventSpan(Span):
    """A consecutive span representing an event. Sometimes we explicitly label e.g. a sentence as the event span.
    label: the event type represented by the event, e.g. Conflict.Attack, CyberAttack, Vulnerability
    """

    def __init__(self, id, int_pair, text, label):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""
        self.sentences = None

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def with_sentences(self, sentences):
        """:type sentences: list[nlplingo.text.text_span.Sentence]"""
        self.sentences = sentences

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def span_type(self):
        return span_types.EVENTSPAN

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))


class EventArgument(Span):
    """A consecutive span representing an event argument
    label: the event argument role, e.g. Source, Target
    """

    def __init__(self, id, entity_mention, label):
        """:type entity_mention: nlplingo.text.text_span.EntityMention"""
        Span.__init__(self, IntPair(entity_mention.start_char_offset(), entity_mention.end_char_offset()), entity_mention.text)
        self.id = id
        self.label = label
        self.entity_mention = entity_mention

    def copy_with_entity_mention(self, entity_mention):
        """Sometimes we want to reassign the entity_mention with one that is backed by tokens
        :type entity_mention: nlplingo.text.text_span.EntityMention"""
        return EventArgument(self.id, entity_mention, self.label)

    def span_type(self):
        return span_types.EVENT_ARGUMENT

    def to_string(self):
        if self.entity_mention.tokens is not None:
            postags = ' '.join(token.pos_tag for token in self.entity_mention.tokens)
        else:
            postags = 'N.A.'
        return (u'%s: %s (%d,%d) "%s" "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, postags, self.label))

    def to_json(self):
        d = dict()
        d['start'] = self.start_char_offset()
        d['end'] = self.end_char_offset()
        d['label'] = self.label
        d['text'] = self.text
        d['mention_id'] = self.entity_mention.id
        return d


class SubwordToken(object):
    """
    There are key differences between a subword token, vs a proper token.
    * A subword token is typically generated from a proper token, which could be a raw token, or a Serif token.
    * For a subword token, we do not store start and end character offsets.
      You typically do not need this information on character offsets, and it might be non trivial to generate.
      Different subword tokenizer might choose to use different prefixes to attach to its subword tokens.
    * Instead, we just store the index of the token, from which this subword token is generated.
    * This class might be extended in future, if we are interested to store embeddings associated with a subword
    """
    def __init__(self, text, source_token_index):
        self.text = text
        self.source_token_index = source_token_index


class Token(Span):
    """An individual word token.
    :spacy_token: an optional spacy token
    """

    def __init__(self, int_pair, index, text, lemma, pos_tag):
        Span.__init__(self, int_pair, text)
        self.index_in_sentence = index     # token index in sentence

        self.lemma = lemma
        self.pos_tag = pos_tag
        self.pos_tag_alternate = None       # for instance, in SerifXML, there is a pos_sequence in each SentenceTheory
        self.dep_relations = []             # dependency relations
        """:type: list[nlplingo.text.dependency_relation.DependencyRelation]"""
        self.child_dep_relations = []
        """:type: list[nlplingo.text.text_span.DependencyRelation]"""

        self.dep_paths_to_root = []
        """:type: list[list[nlplingo.text.text_span.DependencyRelation]]"""
        # might not be completely to root, as we impose a max path length

        # following deals with word embeddings
        self.has_vector = False
        self.vector_index = 0
        self.word_vector = None

        self.srl = None
        """:type: nlplingo.text.text_theory.SRL"""

    def add_dep_relation(self, dep_relation):
        self.dep_relations.append(dep_relation)

    def add_child_dep_relation(self, dep_relation):
        self.child_dep_relations.append(dep_relation)

    def add_dep_path_to_root(self, path):
        self.dep_paths_to_root.append(path)

    def is_punct(self):
        return self.text in punctuations

    def pos_suffix(self):
        if self.pos_tag.startswith('NN') or self.pos_tag.startswith('DET:NN'):
            return '.n'
        elif self.pos_tag.startswith('VB'):
            return '.v'
        elif self.pos_tag.startswith('JJ') or self.pos_tag.startswith('DET:JJ'):
            return '.a'
        else:
            return '.o'

    def text_with_pos_suffix(self):
        return self.text + self.pos_suffix()

    def pos_category(self):
        if self.pos_tag.startswith('NNP') or self.pos_tag.startswith('DET:NNP'):
            return 'PROPN'
        elif self.pos_tag.startswith('NN') or self.pos_tag == 'DET:NN' or self.pos_tag == 'DET:NNS':
            return 'NOUN'
        elif self.pos_tag.startswith('VB') or self.pos_tag == 'VPN':
            return 'VERB'
        elif self.pos_tag.startswith('JJ') or self.pos_tag.startswith('DET:JJ'):
            return 'ADJ'
        else:
            return 'OTHER'

    #def pos_category(self):
    #    tag = self.pos_tag_alternate
    #
    #    ret = 'OTHER'
    #    if tag == 'DET+NOUN+NSUFF_MASC_DU_ACCGEN' or tag == 'DET+NOUN_PRO' or tag == 'DET+NOUN_PROP':
    #        ret = 'OTHER'
    #    elif tag.startswith('DET+NOUN_PROP') or tag.startswith('NOUN_PROP'):
    #        ret = 'PROPN'
    #    elif tag.startswith('DET+NOUN') or tag == 'NOUN' or tag.startswith('NOUN+NSUFF_'):
    #        ret = 'NOUN'
    #    elif tag.startswith('VERB_IMPERATIVE') or tag.startswith('VERB_IMPERFECT') or tag.startswith('VERB_PERFECT'):
    #        ret = 'VERB'
    #    elif '+VERB_IMPERFECT' in tag:
    #        if re.search(r'^FUT\+IV(\S+)\+VERB_IMPERFECT', tag) is not None:
    #            ret = 'VERB'
    #        elif re.search(r'^IV(\S+)\+VERB_IMPERFECT', tag) is not None:
    #            ret = 'VERB'
    #    elif tag.startswith('ADJ') or tag == 'DET+ADJ':
    #        ret = 'ADJ'
    #    else:
    #        ret = 'OTHER'
    #    return ret

    def span_type(self):
        return span_types.TOKEN

    def to_string(self):
        return (u'%s: (%d,%d) "%s"' % (self.span_type(), self.start_char_offset(), self.end_char_offset(), self.text))

    def to_json(self):
        d = dict()
        d['text'] = self.text
        d['index'] = self.index_in_sentence
        d['start'] = self.start_char_offset()
        d['end']  = self.end_char_offset()
        d['lemma'] = self.lemma
        d['pos_tag'] = self.pos_tag
        return d

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start_char_offset() == other.start_char_offset() and self.end_char_offset() == other.end_char_offset()
        return False

    def __ne__(self, other):
        return self.start_char_offset() != other.start_char_offset() or self.end_char_offset() != other.end_char_offset()


class Sentence(Span):
    """Represents a sentence
    :tokens: a list of Token
    """

    def __init__(self, docid, int_pair, text, tokens, index, add_noun_phrases=True):
        """:index: int giving the sentence number (starting from 0) within the document"""
        Span.__init__(self, int_pair, text)
        self.docid = docid
        self.tokens = tokens
        """:type: list[nlplingo.text.text_span.Token]"""
        if add_noun_phrases:
            self.noun_phrases = self._add_noun_phrases()
        else:
            self.noun_phrases = None 
        """:type: list[nlplingo.text.text_span.TextSpan]"""
        self.entity_mentions = []
        """:type: list[nlplingo.text.text_span.EntityMention]"""
        self.events = []
        """:type: list[nlplingo.text.text_theory.Event]"""
        self.srls = []
        """:type: list[nlplingo.text.text_theory.SRL]"""
        self.entity_entity_relations = []
        """:type: list[nlplingo.text.text_theory.EntityRelation]"""
        self.index = index
        self.sent_id = None
        self.vector_index = -1
        self.sent_vector = []
        self.has_vector = False

        # When we are generating subwords, e.g. using Byte Pair Encoding (BPE) or Word Pieces,
        # we want to keep track of which subwords were generated from which original token.
        self.subword_tokens = None
        """:type: list[nlplingo.text.text_span.SubwordToken]"""

    def _get_space_token_offset(self, text, text_start):
        """ Given a sentence text, split on empty spaces, to return list of token character offsets
        :type text: str
        :type text_start: int
        :rtype: list[(int,int, str)]
        """
        start = text_start
        tokens = text.split(' ')    # we use this instead of text.split(), to ensure we get empty tokens for consecutive spaces in text
        offsets = []

        for token in tokens:
            if len(token) == 0:
                start += 1
                continue
            end = start + len(token)
            offsets.append((start, end, token))
            start = end + 1

        return offsets

    def copy_to_space_tokenization(self):
        """ WARNING:
        * This currently only converts the tokens (usually SERIF tokens) to raw space tokens.
        * It leaves unchanged all the other annotations on this sentence, which is non-trivial to change after the fact anyway.

        What is the use case for this?
        * When you only want to do decoding
        * When you are only using the tokens for features
        E.g. when you are feeding the tokens to SentencePiece tokenization, and these are the only features that you use.

        :rtype: nlplingo.text.text_span.Sentence
        """
        offsets = self._get_space_token_offset(self.text, self.start_char_offset())
        new_tokens = []
        for i, offset in enumerate(offsets):
            start = offset[0]
            end = offset[1]
            text = offset[2]
            new_tokens.append(Token(IntPair(start, end), i, text, lemma=None, pos_tag=None))

        sent = Sentence(self.docid, IntPair(self.start_char_offset(), self.end_char_offset()), self.text, new_tokens, self.index, add_noun_phrases=False)
        return sent

    def generate_subword_tokens(self, tokenizer):
        """
        `tokenizer` must be method that is capable of performing tokenization to generate subwords
        """
        for i, token in enumerate(self.tokens):
            subwords = tokenizer.tokenize(token.text)
            for subword in subwords:
                assert type(subword) is str
                self.subword_tokens.append(SubwordToken(subword, i))

    def _add_noun_phrases(self):
        """Now, let's just add all bigrams and trigrams
        """
        ret = []
        """:type: list[nlplingo.text.text_span.TextSpan]"""
        for i in range(len(self.tokens) - 1):  # bigrams
            toks = self.tokens[i:i + 2]
            span = TextSpan(IntPair(toks[0].start_char_offset(), toks[-1].end_char_offset()), ' '.join(t.text for t in toks))
            span.with_tokens(toks)
            ret.append(span)
        for i in range(len(self.tokens) - 2):  # trigrams
            toks = self.tokens[i:i + 3]
            span = TextSpan(IntPair(toks[0].start_char_offset(), toks[-1].end_char_offset()), ' '.join(t.text for t in toks))
            span.with_tokens(toks)
            ret.append(span)
        return ret

    def add_entity_mention(self, entity_mention):
        """:type entity_mention: nlplingo.text.text_span.EntityMention"""
        self.entity_mentions.append(entity_mention)

    def add_event(self, event):
        """:type event: nlplingo.text.text_theory.Event"""
        self.events.append(event)

    def add_srl(self, srl):
        """:type srl: nlplingo.text.text_theory.SRL"""
        self.srls.append(srl)

    def number_of_tokens(self):
        return len(self.tokens)

    def span_type(self):
        return span_types.SENTENCE

    def get_all_event_anchors(self):
        """Returns a list of all event anchors
        Returns:
            list[nlplingo.text.text_span.Anchor]
        """
        # TODO: when reading in events from the annotation files, ensure each token/anchor is only used once
        ret = []
        for event in self.events:
            for anchor in event.anchors:
                ret.append(anchor)
        return ret

    def get_ne_type_per_token(self):
        ret = []
        for token in self.tokens:
            found_ne = False
            for em in self.entity_mentions:
                if em.start_char_offset() <= token.start_char_offset() and token.end_char_offset() <= em.end_char_offset():
                #if token.index_in_sentence == em.head().index_in_sentence:
                    ret.append(em.coarse_label())
                    found_ne = True
                    break
            if not found_ne:
                ret.append('None')
        assert len(ret) == len(self.tokens)
        return ret

    def get_ne_type_with_bio_per_token(self):
        ret = []
        for token in self.tokens:
            found_bio = False
            for em in self.entity_mentions:
                for i, em_token in enumerate(em.tokens):  # type: Token
                    if em_token.index_in_sentence == token.index_in_sentence:
                        if i == 0:
                            label = em.coarse_label() + "_B"
                        else:
                            label = em.coarse_label() + "_I"
                        ret.append(label)
                        found_bio = True
                        break
                if found_bio:
                    break
            if not found_bio:
                ret.append('O')
        assert len(ret) == len(self.tokens)
        return ret

    def get_text(self, start, end):
        """If the given start, end character offsets are within this sentence, return the associated text"""
        if self.start_char_offset() <= start:
            if end <= self.end_char_offset():
                normalized_start = start - self.start_char_offset()
                normalized_end = normalized_start + (end - start)
                # print('Returning {}-{} from "{}"'.format(normalized_start, normalized_end, self.text))
                return self.text[normalized_start:normalized_end]
            elif start <= self.end_char_offset():
                normalized_start = start - self.start_char_offset()
                normalized_end = normalized_start + (end - start)
                logger.warning(
                    'Span {}-{} extends beyond sentence (available: "{}")'
                    .format(start, end, self.text[normalized_start:normalized_end]))
            else:
                pass
        else:
            if self.start_char_offset() <= end <= self.end_char_offset():
                normalized_start = start - self.start_char_offset()
                normalized_end = normalized_start + (end - start)
                logger.warning(
                    'Span {}-{} overlaps with start of sentence (available: "{}")'
                    .format(start, end, self.text[:normalized_end]))
            else:
                pass
        # logger.debug('sentence: {}-{}, span: {}-{}'.format(self.start_char_offset(), self.end_char_offset(), start, end))#span: {}-{}')
        return None

    def to_string(self):
        return u'(%d,%d):[%s]' % (self.start_char_offset(), self.end_char_offset(), self.text)

    def tokens_as_json(self):
        d = dict()
        d['tokens'] = [token.to_json() for token in self.tokens]
        return d

    def mentions_as_json(self):
        d = dict()
        d['mentions'] = [mention.to_json() for mention in self.entity_mentions]
        return d

    def events_as_json(self):
        d = dict()
        d['events'] = [event.to_json() for event in self.events]
        return d

    def relations_as_json(self):
        d = dict()
        d['relations'] = [relation.to_json() for relation in self.entity_entity_relations]
        return d

    def word_embedding(self, length, PAD_IDX=0):
        """
        Return a length-sized numpy array with token embedding indices inside of it.
        Capture the embeddings index, at each word position in sentence
        :param length:
        :param PAD_IDX:
        :return:
        """
        rtn = np.empty(length, dtype=int_type)
        rtn.fill(PAD_IDX)
        for i, token in enumerate(self.tokens):
            rtn[i] = token.vector_index
        return rtn

    # NER with BIO of each token
    def entity_types(self, length, event_domain, PAD_IDX=0):
        """
        Returns a length-sized numpy array with entity type indices defined by an event_domain.
        :param length: int
        :param PAD_IDX: int
        :return:
        """
        rtn = np.empty(length, dtype=int_type)
        rtn.fill(PAD_IDX)
        token_ne_bio_type = self.get_ne_type_with_bio_per_token()
        assert len(token_ne_bio_type) == len(self.tokens)
        for i, token in enumerate(self.tokens):
            rtn[i] = event_domain.get_entity_bio_type_index(token_ne_bio_type[i])
        return rtn

    def word_embedding_vector(self, length, embedding_vector_size, PAD_IDX=0):
        """
        Returns a length x embedding_vector_size numpy array.
        Capture the actual word embeddings, at each word position in sentence
        :param length: int
        :param embedding_vector_size: int
        :param PAD_IDX: int
        :return:
        """
        rtn = np.empty((length, embedding_vector_size), dtype=float_type)
        rtn.fill(PAD_IDX)
        for i, token in enumerate(self.tokens):
            if token.word_vector is not None:
                rtn[i, :] = token.word_vector
        return rtn

#
# class DependencyRelation(object):
#     def __init__(self, dep_name, parent_token_index, child_token_index):
#         self.dep_name = dep_name
#         self.parent_token_index = parent_token_index
#         self.child_token_index = child_token_index
#
#     def to_string(self):
#         return '{}:{}:{}'.format(self.dep_name, self.parent_token_index, self.child_token_index)

def to_sentence(text, start, end):
    """Converts a sentence raw text to a Sentence object."""
    charOffsets = IntPair(start, end)
    tokens = []

    offset = start
    for i, t in enumerate(text.split()):
        token = Token(IntPair(offset, offset+len(t)), i, t, lemma=None, pos_tag=None)
        tokens.append(token)
        offset += len(t)+1    # +1 to account for white-space

    return Sentence('docid', charOffsets, text, tokens, 0)


def file_to_document(filepath):
    f = open(filepath, 'rU')
    sentences = []

    offset = 0
    for line in f:
        sentence = to_sentence(line, offset, offset+len(line))
        sentences.append(sentence)
        offset += len(line);    # +1 for account for newline
    f.close()

    s_strings = [s.label for s in sentences]
    doc_text = "\n".join(s_strings)

    return Document(IntPair(0, offset-1), doc_text, sentences)


def spans_overlap(span1, span2):
    """
    :type span1: nlplingo.text.text_span.Span
    :type span2: nlplingo.text.text_span.Span
    """
    start1 = span1.start_char_offset()
    end1 = span1.end_char_offset()
    start2 = span2.start_char_offset()
    end2 = span2.end_char_offset()

    if start1 != start2 and end1 != end2 and (end1 <= start2 or end2 <= start1):
        return False
    else:
        return True

