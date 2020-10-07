
from collections import defaultdict
import logging
import sys

from nlplingo.common.io_utils import safeprint
from nlplingo.common.span_utils import get_tokens_corresponding_to_span, get_tokens_covering_span
from nlplingo.common.utils import IntPair
from nlplingo.common.span_utils import get_span_with_offsets
from nlplingo.common.span_utils import get_spans_in_offsets
from nlplingo.text.text_span import Token, EntityMention
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import spans_overlap
from nlplingo.text.dependency_relation import DependencyRelation

class Entity(object):
    def __init__(self, entity_id):
        self.mentions = []
        self.id = entity_id

class Document(object):
    """Represents a document
    """
    verbosity = 1

    def __init__(self, docid, text=None, sentence_strings=None):
        self.docid = docid
        self.text = text
        self.sentence_strings = sentence_strings    # sometimes you want to keep the strings separate, e.g. for ACE
        """:type: list[str]"""
        self.sentences = []
        """:type: list[nlplingo.text.text_span.Sentence]"""
        self.entity_mentions = []
        """:type: list[nlplingo.text.text_span.EntityMention]"""
        self.events = []
        """:type: list[nlplingo.text.text_theory.Event]"""
        self.coref_events = []
        """:type list[list[nlplingo.text.text_theory.Event]]"""
        self.entities_by_id = dict()
        self.event_event_relations = []
        self.entity_entity_relations = []
        self.adj_mats = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def add_entity_mention(self, entity_mention):
        self.entity_mentions.append(entity_mention)

    def add_entity(self, entity):
        self.entities_by_id[entity.id] = entity

    def add_event(self, event):
        self.events.append(event)

    def add_coref_events(self, events):
        self.coref_events.append(events)

    def add_event_event_relation(self, event_event_relation):
        self.event_event_relations.append(event_event_relation)

    def add_entity_entity_relation(self, entity_entity_relation):
        self.entity_entity_relations.append(entity_entity_relation)

    def add_adj_mat(self, adj_mat):
        self.adj_mats.append(adj_mat)

    def number_of_entity_mentions(self):
        return len(self.entity_mentions)

    def number_of_events(self):
        return len(self.events)

    def annotate_sentences(self, embeddings, model, fuzzy_match_entitymention=False, fuzzy_match_anchor=False):
        """We use Spacy for model
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        #print('Let us first show the event annotations in this document')
        #for event in self.events:
        #    print(event.to_string())
        #print('==============')

        # Sometimes, sentence splitting and tokenization has already been done, e.g. using corenlp
        if model is not None:
            self.sentence_segmention_and_tokenization(model)

        for sent in self.sentences:
            if len(embeddings) > 0:
                self.annotate_sentence_with_word_embeddings(sent, embeddings)
            self.annotate_sentence_with_entity_mentions(sent, fuzzy_match_entitymention)
            self.annotate_sentence_with_events(sent, fuzzy_match_anchor)
            self.annotate_sentence_with_entity_relations(sent)
            #print(sent.text)
            #print('===Events===')
            #for event in sent.events:
            #    print(event.to_string())
        #exit(0)

        # print('==== Finished annotating sentences, printing from within text_theory.Document.annotate_sentences ====')
        # print('#document events={}'.format(len(self.events)))
        # for sent in self.sentences:
        #     sent.to_string()
        #     for e in sent.events:
        #         print('<Sentence-Events>')
        #         print(e.to_string())
        #         print('</Sentence-Events>')
        #     print('')
        # exit(0)


    def sentence_segmention_and_tokenization(self, model):
        """Whatever model we pass in, must be able to perform sentence segmentation and tokenization
        by calling model(self.text). We typically use Spacy
        """
        if self.text is not None:
            self.sentence_segmention_and_tokenization_with_text(model)
        elif self.sentence_strings is not None:
            self.sentence_segmention_and_tokenization_with_list(model)
        else:
            print('ERROR: text and sentence_strings are both None, this should not happen')

    def sentence_segmention_and_tokenization_with_text(self, model):
        """Whatever model we pass in, must be able to perform sentence segmentation and tokenization
        by calling model(self.text). We typically use Spacy
        """
        doc = model(self.text)

        for sent_index, sent in enumerate(doc.sents):
            tokens = []
            for token_index, token in enumerate(sent):
                start = token.idx
                end = token.idx + len(token.text)
                tokens.append(Token(IntPair(start, end), token_index, token.text, token.lemma_, token.tag_))
            sentence = Sentence(self.docid, IntPair(sent.start_char, sent.end_char), sent.text.strip(), tokens, sent_index)
            self.sentences.append(sentence)

    def sentence_segmention_and_tokenization_with_list(self, model):
        """Whatever model we pass in, must be able to perform sentence segmentation and tokenization
        by calling model(self.text). We typically use Spacy
        """
        offset = 0
        for ss in self.sentence_strings:
            if len(ss) == 0 or ss.isspace():
                pass
            else:
                for sent in model(ss).sents:  # for each Spacy sentence
                    tokens = []
                    for token_index, token in enumerate(sent):
                        start = offset + token.idx
                        end = start + len(token.text)
                        tokens.append(Token(IntPair(start, end), token_index, token.text, token.lemma_, token.tag_))
                    sentence = Sentence(self.docid,
                                        IntPair(offset + sent.start_char, offset + sent.start_char + len(sent.text)),
                                        sent.text.strip(), tokens, len(self.sentences))
                    self.sentences.append(sentence)
            offset += len(ss)

    def annotate_sentence_with_word_embeddings(self, sent, embeddings, embedding_prefix=None):
        """Annotate sentence tokens with word embeddings
        :type sent: nlplingo.text.text_span.Sentence
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbeddingAbstractt
        """
        def convert_dep_relname(dep_rel_name):
            import re
            compound_match = re.match('nmod\:\w+_(\w+)', dep_rel_name)
            if compound_match:
                dep_rel_name = u'adpmod:' + compound_match.group(1)
            if dep_rel_name == 'nmod:poss':
                return 'poss'
            elif dep_rel_name == 'nmod:tmod':
                return 'advmod'
            elif dep_rel_name == 'nmod:agent':
                return 'adpmod:by'
            elif dep_rel_name.startswith('nmod:'):
                return ':'.join(['adpmod', dep_rel_name.split(':')[1]])
            elif dep_rel_name == 'nsubj:xsubj':
                return 'nsubj'
            elif dep_rel_name == 'nsubjpass:xsubj':
                return 'nsubjpass'
            elif dep_rel_name == 'compound':
                return 'compmod'
            elif dep_rel_name == 'nummod':
                return 'num'
            elif dep_rel_name.startswith('advcl:'):
                return 'advcl'
            elif dep_rel_name.startswith('conj:'):
                return 'conj'
            else:
                return dep_rel_name
        sent_index, sent_vector = embeddings['word_embeddings'].get_sent_vector(sent)
        sent.vector_index = sent_index
        sent.sent_vector = sent_vector
        sent.has_vector = True

        for token in sent.tokens:
            token.dep_rel_index_lookup = dict()
            if token.is_punct() is False:
                vector_text, index, vector = embeddings['word_embeddings'].get_vector(token, embedding_prefix, sent=sent)
                if vector_text:
                    token.vector_text = vector_text
                    token.vector_index = index
                    #token.word_vector = vector
                    token.has_vector = True
                    ##
                    if 'dependency_embeddings' in embeddings:
                        for dep_rel in token.dep_relations:
                            modifier_text = dep_rel.dep_name
                            target_text = sent.tokens[dep_rel.connecting_token_index].text
                            direction_modifier = 'I' if dep_rel.dep_direction == 'UP' else ''
                            key = modifier_text + direction_modifier + '_' + target_text
                            mkey = convert_dep_relname(modifier_text) + direction_modifier + '_' + target_text.lower()
                            vector_text, index, vector = embeddings['dependency_embeddings'].get_vector(mkey)
                            #if index == -1:
                               # print(u'Missed: {}'.format(key))
                            if index != -1:
                                token.dep_rel_index_lookup[key] = index
                    ##

                else:
                    token.vector_index = embeddings['word_embeddings'].missing_index
            else:
                token.vector_index = embeddings['word_embeddings'].missing_index

    def annotate_sentence_with_entity_mentions(self, sent, fuzzy_match=False):
        """We assign entity mentions that are properly backed by tokens to the sentence
        :type sent: nlplingo.text.text_span.Sentence
        """

        # get the list of EntityMention in this sentence
        sent_em = self.get_entity_mentions_in_span(sent.start_char_offset(), sent.end_char_offset())
        for em in sent_em:
            matched_tokens = get_tokens_corresponding_to_span(sent.tokens, em)
            if matched_tokens is not None:
                em.with_tokens(matched_tokens)
                sent.add_entity_mention(em)
            else:
                covering_tokens = get_tokens_covering_span(sent.tokens, em)
                if covering_tokens is not None:
                    if fuzzy_match:
                        em.with_tokens(covering_tokens)
                        sent.add_entity_mention(em)
                    else:   # we do require exact match on the start_char_offset() and end_char_offset()
                        logging.debug(
                            u'EntityMention not backed by tokens, best-matching-tokens="{}":{}-{}, dropping docid={} em={}'.format(
                                u' '.join(t.text for t in covering_tokens), covering_tokens[0].start_char_offset(),
                                covering_tokens[-1].end_char_offset(), sent.docid, em.to_string()))
                else:
                    logging.debug(u'EntityMention not backed by tokens, covering_tokens=None, dropping docid={} em={}'.format(sent.docid, em.to_string()))


    def annotate_sentence_with_entity_relations(self, sent):
        """
        :type sent: nlplingo.text.text_span.Sentence
        """
        for relation in self.entity_entity_relations:
            arg1 = relation.arg1
            """:type: nlplingo.text.text_span.EntityMention"""
            arg2 = relation.arg2
            """:type: nlplingo.text.text_span.EntityMention"""

            if sent.contains(arg1) and sent.contains(arg2):     # based on char offsets
                em1 = get_span_with_offsets(sent.entity_mentions, arg1.start_char_offset(), arg1.end_char_offset())
                if em1 is None:
                    safeprint(u'RelationArgument not backed by EntityMention, dropping docid={} arg={}'.format(sent.docid, arg1.to_string()))

                em2 = get_span_with_offsets(sent.entity_mentions, arg2.start_char_offset(), arg2.end_char_offset())
                if em2 is None:
                    safeprint(u'RelationArgument not backed by EntityMention, dropping docid={} arg={}'.format(sent.docid, arg2.to_string()))

                if em1 is not None and em2 is not None:
                    sent.entity_entity_relations.append(relation)


    def annotate_sentence_with_events(self, sent, anchor_fuzzy_match=False):
        """For an event, anchors must be backed by tokens, and arguments must be backed by entity mentions
        :type sent: nlplingo.text.text_span.Sentence
        """
        event_counter = 0
        for event in self.events:
            # for this document-level 'event', we first check whether it has anchors/arguments in this sentence
            anchors = event.get_anchors_in_span(sent.start_char_offset(), sent.end_char_offset())
            valid_anchors = []
            for anchor in anchors:
                matched_tokens = get_tokens_corresponding_to_span(sent.tokens, anchor)
                if matched_tokens is not None:
                    anchor.with_tokens(matched_tokens)
                    valid_anchors.append(anchor)
                else:
                    covering_tokens = get_tokens_covering_span(sent.tokens, anchor)
                    if covering_tokens is not None:
                        if anchor_fuzzy_match:
                            anchor.with_tokens(covering_tokens)
                            valid_anchors.append(anchor)
                        else:  # we do require exact match on the start_char_offset() and end_char_offset()
                            logging.debug(u'Anchor not backed by tokens, best-matching-tokens="{}":{}-{}, dropping docid={} anchor={}'.format(
                                u' '.join(t.text for t in covering_tokens), covering_tokens[0].start_char_offset(),
                                covering_tokens[-1].end_char_offset(), sent.docid, anchor.to_string()))
                    else:
                        logging.debug(u'Anchor not backed by tokens, covering_tokens=None, dropping docid={} anchor={}'.format(sent.docid, anchor.to_string()))

            if len(valid_anchors) == 0:
                continue

            arguments = event.get_arguments_in_span(sent.start_char_offset(), sent.end_char_offset())
            valid_arguments = []
            for arg in arguments:
                em = get_span_with_offsets(sent.entity_mentions, arg.start_char_offset(), arg.end_char_offset())
                if em is not None:  # there is an EntityMention corresponding to the EventArgument
                    valid_arguments.append(arg.copy_with_entity_mention(em))
                else:
                    safeprint(u'EventArgument not backed by EntityMention, dropping docid={} arg={}'.format(sent.docid, arg.to_string()))

            event_id = self.docid + '-s' + str(sent.index) + '-e' + str(event_counter)
            sent_event = Event(event_id, event.label)
            sent_event.add_anchors(valid_anchors)
            if len(valid_arguments) > 0:
                sent_event.add_arguments(valid_arguments)
            sent.add_event(sent_event)
            event_counter += 1

    def get_entity_mention_with_span(self, start, end):
        """Get the entity mention with the exact same given (start, end) span
        Returns:
            nlplingo.text.text_span.EntityMention
        """
        return get_span_with_offsets(self.entity_mentions, start, end)


    def get_entity_mentions_in_span(self, start, end):
        """Useful for getting all entity mentions in a sentence
        Returns:
            list[nlplingo.text.text_span.EntityMention]
        """
        return get_spans_in_offsets(self.entity_mentions, start, end)

    def get_entity_mention_with_id(self, id):
        for em in self.entity_mentions:
            if em.id == id:
                return em
        return None

    def get_event_with_id(self, id):
        for e in self.events:
            if e.id == id:
                return e
        return None

    def get_text(self, start, end):
        """Given start, end character offsets, return the text associated.
        If self.text is defined, then directly use that to return the text.
        Else, go through each sentence, and use the Sentence.get_text(start, end)
        """
        if self.text is not None:
            return self.text[start:end]
        else:
            #print('text_theory.Document {}, trying to find text for {}-{}'.format(self.docid, start, end))
            for sentence in self.sentences:
                #print('Checking against sentence ({}-{})'.format(sentence.start_char_offset(), sentence.end_char_offset()))
                text = sentence.get_text(start, end)
                if text is not None:
                    #print('Found text "{}" in sentence {}-{}'.format(text, sentence.start_char_offset(), sentence.end_char_offset()))
                    return text
        return None

    def apply_domain(self, event_domain):
        """
        :type event_domain: nlplingo.tasks.event_domain.EventDomain
        """
        self.events = [event for event in self.events if event.label in event_domain.event_type_mappings]
        for sent in self.sentences:
            sent.events = [event for event in sent.events if event.label in event_domain.event_type_mappings]

        for event in self.events:
            event.apply_domain(event_domain)
        for sent in self.sentences:
            for event in sent.events:
                event.apply_domain(event_domain)

    def text_length(self):
        if self.text is not None:
            return len(self.text)
        elif len(self.sentences) > 0:
            return self.sentences[-1].end_char_offset()
        else:
            return 0

    def to_string(self):
        outlines = []
        prefix = ('<DOC docid=%s>\n<TEXT>\n%s\n</TEXT>' % (self.docid, self.text))
        outlines.append(prefix)
        for sent in self.sentences:
            outlines.append('<Sentence index=%d text="%s">' % (sent.index, sent.text))
            outlines.append('TOKENS: %s' % (' '.join(token.text for token in sent.tokens)))
            for em in sent.entity_mentions:
                outlines.append(em.to_string())
            for e in sent.events:
                outlines.append(e.to_string())
        outlines.append('</DOC>')
        return '\n'.join(outlines)

    def to_json(self):
        doc_d = dict()
        doc_d['docid'] = self.docid
        doc_d['text'] = self.text
        sentence_dicts = []
        for sentence in self.sentences:
            sentence_d = dict()
            sentence_d['index'] = sentence.index
            sentence_d['text'] = sentence.text
            sentence_d['start'] = sentence.start_char_offset()
            sentence_d['end'] = sentence.end_char_offset()

            sentence_d.update(sentence.tokens_as_json())

            sentence_d.update(sentence.mentions_as_json())

            sentence_d.update(sentence.events_as_json())

            sentence_d.update(sentence.relations_as_json())

            # token_dicts = []
            # for token in sentence.tokens:
            #     token_d = dict()
            #     token_d['index'] = token.index_in_sentence
            #     token_d['text'] = token.text
            #     token_d['start'] = token.start_char_offset()
            #     token_d['end'] = token.end_char_offset()
            #     token_d['lemma'] = token.lemma
            #     token_d['pos_tag'] = token.pos_tag
            #
            #     dep_dicts = []
            #     for r in token.dep_relations:
            #         dep_d = dict()
            #         dep_d['dep_name'] = r.dep_name
            #         dep_d['dep_token_index'] = r.connecting_token_index
            #         dep_d['dep_token_text'] = sentence.tokens[r.connecting_token_index].text
            #         dep_d['dep_direction'] = r.dep_direction
            #         dep_dicts.append(dep_d)
            #     token_d['dep_relations'] = dep_dicts
            #
            #     srl_dict = dict()
            #     if token.srl is not None:
            #         srl = token.srl
            #         srl_dict['predicate'] = srl.predicate_label
            #         srl_role_dicts = []
            #         for role in srl.roles:
            #             for start_index, end_index in srl.roles[role]:
            #                 d = dict()
            #                 d['srl_role'] = role
            #                 d['srl_token_span'] = [start_index, end_index]
            #                 d['srl_token_text'] = ' '.join(sentence.tokens[index].text for index in range(start_index, end_index+1))
            #                 #d['srl_token_text'] = sentence.tokens[index].text
            #                 srl_role_dicts.append(d)
            #         srl_dict['roles'] = srl_role_dicts
            #     token_d['srl'] = srl_dict
            #
            #     token_dicts.append(token_d)
            # sentence_d['tokens'] = token_dicts

            sentence_dicts.append(sentence_d)
        doc_d['sentences'] = sentence_dicts
        return doc_d

    @staticmethod
    def from_json(doc_d):
        docid = doc_d['docid']
        doc = Document(docid, doc_d['text'])

        for sentence_d in doc_d['sentences']:
            sentence_index = sentence_d['index']
            sentence_text = sentence_d['text']
            sentence_start = sentence_d['start']
            sentence_end = sentence_d['end']
            sentence_mentions = sentence_d.get('mentions', [])

            tokens = []
            """:type: list[nlplingo.text.text_span.Token]"""
            for token_d in sentence_d['tokens']:
                index = token_d['index']
                text = token_d['text']
                start = token_d['start']
                end = token_d['end']
                lemma = token_d['lemma']
                pos_tag = token_d['pos_tag']
                token = Token(IntPair(start, end), index, text, lemma, pos_tag)

                for dep_d in token_d.get('dep_relations', []):
                    name = dep_d['dep_name']
                    direction = dep_d['dep_direction']
                    index = dep_d['dep_token_index']
                    token.dep_relations.append(DependencyRelation(name, direction, index))

                srl_dict = token_d.get('srl', {})
                if 'predicate' in srl_dict:
                    srl = SRL('dummy')
                    srl.predicate_label = srl_dict['predicate']
                    if 'roles' in srl_dict:
                        for role_d in srl_dict['roles']:
                            role = role_d['srl_role']
                            token_span = role_d['srl_token_span']   # list of 2 int: start-token-index, end-token-index
                            srl.add_role(role, token_span[0], token_span[1])
                    token.srl = srl
                tokens.append(token)

            sentence = Sentence(docid, IntPair(sentence_start, sentence_end), sentence_text, tokens, sentence_index)

            for m in sentence_mentions:
                em = EntityMention(m['id'], IntPair(m['start'], m['end']), m['text'], m['label'])
                em_tokens = []
                for t in m['tokens']:
                    em_tokens.append(Token(IntPair(t['start'], t['end']), t['index'], t['text'], t['lemma'], t['pos_tag']))
                em.with_tokens(em_tokens)
                sentence.add_entity_mention(em)

            doc.add_sentence(sentence)
        return doc


class Event(object):
    """Annotation of an event. There is no explicit differentiation between event and event mention
    label: event type
    anchors: a list of nlplingo.text.text_span.Anchor , you can think of each anchor as an event mention
    arguments: a list of nlplingo.text_span.EventArgument
    """

    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.anchors = []
        """:type: list[nlplingo.text.text_span.Anchor]"""
        self.arguments = []
        """:type: list[nlplingo.text.text_span.EventArgument]"""
        self.event_spans = []
        """:type: list[nlplingo.text.text_span.EventSpan]"""

    def add_anchor(self, anchor):
        self.anchors.append(anchor)

    def add_anchors(self, anchors):
        self.anchors.extend(anchors)

    def add_event_span(self, span):
        self.event_spans.append(span)

    def number_of_anchors(self):
        return len(self.anchors)

    def overlaps_with_anchor(self, span):
        """Check whether input span overlaps with any of our anchors
        :type anchor: nlplingo.text.text_span.Span
        """
        for a in self.anchors:
            if spans_overlap(span, a):
                return True
        return False

    def add_argument(self, argument):
        self.arguments.append(argument)

    def add_arguments(self, arguments):
        self.arguments.extend(arguments)

    def number_of_arguments(self):
        return len(self.arguments)

    def get_anchors_in_span(self, start, end):
        """Get the list of anchors within a given (start, end) character offset, e.g. within a sentence
        Returns:
            list[nlplingo.text.text_span.Anchor]
        """
        return get_spans_in_offsets(self.anchors, start, end)

    def get_arguments_in_span(self, start, end):
        """Get the list of arguments within a given (start, end) character offset, e.g. within a sentence
        Returns:
            list[nlplingo.text.text_span.EventArgument]
        """
        return get_spans_in_offsets(self.arguments, start, end)

    def event_from_span(self, start, end):
        """Generate a new event from given (start, end) character offsets, e.g. within a sentence
        Returns:
            nlplingo.text.text_theory.Event
        """
        event = Event(self.id, self.label)
        event.add_anchors(self.get_anchors_in_span(start, end))
        event.add_arguments(self.get_arguments_in_span(start, end))
        return event

    def get_role_for_entity_mention(self, entity_mention):
        """
        :type entity_mention: nlplingo.text.text_span.EntityMention
        """
        for arg in self.arguments:
            if entity_mention == arg.entity_mention:
                return arg.label
        return 'None'

    def apply_domain(self, event_domain):
        """:type event_domain: nlplingo.tasks.event_domain.EventDomain"""
        self.label = event_domain.event_type_mappings[self.label]

        new_args = [arg for arg in self.arguments if event_domain.event_type_role_in_domain(self.label, arg.label)]

        for arg in new_args:
            for role in event_domain.event_type_role[self.label]:
                if role.label == arg.label:
                    arg.label = role.map_to
                    break

        for anchor in self.anchors:
            anchor.label = self.label

        self.arguments = new_args

    def to_string(self):
        anchor_strings = '\n'.join(a.to_string() for a in self.anchors)
        argument_strings = '\n'.join(a.to_string() for a in self.arguments)
        span_strings = '\n'.join(a.to_string() for a in self.event_spans)
        prefix = ('<Event id=%s type=%s>' % (self.id, self.label))
        return prefix + '\n#anchors=' + str(len(self.anchors)) + ' #args=' + str(len(self.arguments)) + '\n' + \
               span_strings + '\n' + anchor_strings + '\n' + argument_strings + '\n</Event>'

    def to_json(self):
        d = dict()
        d['label'] = self.label
        d['anchors'] = [anchor.to_json() for anchor in self.anchors]
        d['arguments'] = [argument.to_json() for argument in self.arguments]
        return d


class EventEventRelation(object):
    def __init__(self, type, arg1, arg2, serif_sentence=None, serif_event_0=None, serif_event_1=None):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2

        self.serif_sentence = serif_sentence
        self.serif_event_0 = serif_event_0
        self.serif_event_1 = serif_event_1


class EntityRelation(object):
    def __init__(self, type, arg1, arg2):
        self.type = type
        self.arg1 = arg1
        """:type: nlplingo.text.text_span.EntityMention"""
        self.arg2 = arg2
        """:type: nlplingo.text.text_span.EntityMention"""

    def to_json(self):
        d = dict()
        d['label'] = self.type
        d['arg1'] = self.arg1.to_json()
        d['arg2'] = self.arg2.to_json()
        return d


class SRL(object):
    def __init__(self, id):
        self.id = id
        #self.predicate_token = None
        #""":type: nlplingo.text.text_span.Token"""
        self.predicate_label = None
        #self.roles = defaultdict(list)     # we do a list instead of dict, because there could be multiple spans of the same role
        #""":type: dict[str, list[nlplingo.text.text_span.TextSpan]]"""
        self.roles = defaultdict(list)  # there could be multiple spans of the same role


    def add_role(self, role, start_token_index, end_token_index):
        """
        :type role: str
        """
        self.roles[role].append((start_token_index, end_token_index))


def annotate_sentence_with_word_embeddings(sent, embeddings, embedding_prefix=None):
    """
    Annotate sentence tokens with word embeddings.
    This function was taken out of Document in order for usage on free-standing sentences (not within a nlplingo Document).
    :type sent: nlplingo.text.text_span.Sentence
    :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbeddingAbstract
    """
    for token in sent.tokens:
        if token.is_punct() is False:
            vector_text, index, vector = embeddings['word_embeddings'].get_vector(token, embedding_prefix)
            if vector_text:
                token.vector_text = vector_text
                token.vector_index = index
                token.has_vector = True
            else:
                token.vector_index = embeddings['word_embeddings'].missing_index
        else:
            token.vector_index = embeddings['word_embeddings'].missing_index
