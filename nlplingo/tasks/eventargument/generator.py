
import codecs
from collections import defaultdict

from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import trigger_pos_category
from nlplingo.tasks.eventargument.example import EventArgumentExample
from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.tasks.common.unary.event_within_sentence import EventWithinSentence
from nlplingo.tasks.common.unary.entity_within_sentence import EntityWithinSentence

class EventArgumentExampleGenerator(ExampleGenerator):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class to generate candidate Datapoint objects for the event argument
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EventArgumentExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)

        self.filter_using_entity_type = self.params.get('argument_generator.filter_using_entity_type', False)

        if 'trigger_argument_span' in self.params:
            self.trigger_argument_span = self.read_trigger_argument_spanfile(self.params['trigger_argument_span'])
        else:
            self.trigger_argument_span = None   # docid -> [ (trigger_offset, argument_offset), ... ]

        """
        TODO: fix graph models 
        self.adjGraphOn = False
        if ('adj_graph' in extractor_params['features']) or ('head_array' in extractor_params['features']):
            self.adjGraphOn = True

        self.trigger_arg_variable = False
        if ('trigger_word_position_variable' in extractor_params['features']) and ('argument_word_position_variable' in extractor_params['features']):
            self.trigger_arg_variable = True
        """

    def read_trigger_argument_spanfile(self, filepath):
        ret = defaultdict(list)
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                docid = tokens[0]
                trigger_offset = IntPair(int(tokens[1]), int(tokens[2]))
                argument_offset = IntPair(int(tokens[3]), int(tokens[4]))
                ret[docid].append((trigger_offset, argument_offset))
        return ret

    def generate(self, docs, triggers=None):
        """
        The current event-argument generation scheme assumes the event trigger and entity argument are both in the same sentence.
        +1
        :type docs: list[nlplingo.text.text_theory.Document]
        :type triggers: defaultdict(list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint])
        :type chunk_writer: nlplingo.common.serialize_disk.ChunkWriter
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""

        for doc in docs:
            if triggers is not None:
                doc_triggers = triggers[doc.docid]
                """:type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""

                # organize the doc_triggers by sentence number
                sent_triggers = defaultdict(list)
                for trigger in doc_triggers:
                    sent_triggers[trigger.sentence.index].append(trigger)

                for sent in doc.sentences:
                    chunk = self.filter_examples(self._generate_sentence(sent, trigger_egs=sent_triggers[sent.index]))
                    """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
                    examples.extend(chunk)
            else:
                for sent in doc.sentences:
                    chunk = self.filter_examples(self._generate_sentence(sent))
                    """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
                    examples.extend(chunk)

        for k, v in self.statistics.items():
            print('EventArgumentExampleGenerator stats, {}:{}'.format(k, v))

        return examples

    def filter_examples(self, argument_examples):
        """
        :type argument_examples: list[nlplingo.tasks.eventargument.example.EventArgumentExample]
        """
        arg_examples = argument_examples
        """
        if self.trigger_arg_variable:
            arg_examples = []
            for eg in argument_examples:
                subj_pos = [i for i in range(len(eg.trigger_pos_data)) if eg.trigger_pos_data[i] == 0]
                obj_pos = [i for i in range(len(eg.argument_pos_data)) if eg.argument_pos_data[i] == 0]
                if subj_pos != obj_pos:
                    arg_examples.append(eg)
        """

        if self.trigger_argument_span is None:
            return arg_examples

        ret = []
        for eg in arg_examples:
            docid = eg.sentence.docid
            use_example = False
            if docid in self.trigger_argument_span:
                for (trigger_offset, argument_offset) in self.trigger_argument_span[docid]:
                    if eg.anchor.start_char_offset() == trigger_offset.first and \
                            eg.anchor.end_char_offset() == trigger_offset.second and \
                            eg.argument.start_char_offset() == argument_offset.first and \
                            eg.argument.end_char_offset() == argument_offset.second:
                        use_example = True
                        break
            if use_example:
                ret.append(eg)
        return ret


    @staticmethod
    def get_event_role(anchor, entity_mention, events):
        """
        +1
        If the given (anchor, entity_mention) is found in the given events, return role label, else return 'None'
        :type anchor: nlplingo.text.text_span.Anchor
        :type entity_mention: nlplingo.text.text_span.EntityMention
        :type events: list[nlplingo.text.text_theory.Event]
        """

        for event in events:
            for a in event.anchors:
                if anchor.start_char_offset() == a.start_char_offset() and anchor.end_char_offset() == a.end_char_offset():
                    role = event.get_role_for_entity_mention(entity_mention)
                    if role != 'None':
                        return role
        return 'None'

    def _generate_sentence(self, sentence, trigger_egs=None, adj_mat=None):
        """
        +1
        We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator
        :type trigger_egs: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        """
        # skip multi-token triggers, args that do not have embeddings, args that overlap with trigger
        ret = []
        """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
        embedding_vector_size = self.extractor_params['embeddings']['vector_size']
        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.hyper_params.max_sentence_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        if trigger_egs is not None:
            for trigger_index, eg in enumerate(trigger_egs):
                anchor_id = '{}-s{}-t{}'.format(sentence.docid, sentence.index, trigger_index)
                anchor = Anchor(anchor_id, IntPair(eg.anchor.start_char_offset(), eg.anchor.end_char_offset()),
                                eg.anchor.text, eg.event_type)
                anchor.with_tokens(eg.anchor.tokens)

                for em in sentence.entity_mentions:
                    role = 'None'
                    if not self.filter_using_entity_type or (
                            self.filter_using_entity_type and em.coarse_label() in self.event_domain.entity_types.keys()):

                        e1 = EventWithinSentence(anchor, self.event_domain, embedding_vector_size, anchor.label, sentence, head_only=True)
                        e2 = EntityWithinSentence(em, self.event_domain, embedding_vector_size, None, sentence)
                        example = EventArgumentExample(e1, e2, self.event_domain, role)
                        ret.append(example)
        else:
            self.statistics['#Sentence-Events'] += len(sentence.events)
            self.statistics['#Sentence-Entity_Mentions'] += len(sentence.entity_mentions)
            for event in sentence.events:
                for anchor in event.anchors:
                    self.statistics['#Sentence-Anchors'] += 1
                    if anchor.head().pos_category() in trigger_pos_category:
                        for em in sentence.entity_mentions:
                            role = event.get_role_for_entity_mention(em)
                            self.statistics['#Event-Role {}'.format(role)] += 1
                            # if spans_overlap(anchor, em):
                            #     print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
                            # else:
                            #     if role != 'None':
                            #         self.statistics['number_positive_argument'] += 1
                            #     example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
                            #     self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                            #     ret.append(example)
                            if role != 'None':
                                self.statistics['number_positive_argument'] += 1
                                self.statistics['TP argument coarse_label={}'.format(em.coarse_label())] += 1

                            if not self.filter_using_entity_type or (
                                    self.filter_using_entity_type and em.coarse_label() in self.event_domain.entity_types.keys()):
                                e1 = EventWithinSentence(anchor, self.event_domain, embedding_vector_size, anchor.label,
                                                         sentence, head_only=True)
                                e2 = EntityWithinSentence(em, self.event_domain, embedding_vector_size, None, sentence)
                                example = EventArgumentExample(e1, e2, self.event_domain, role)
                                ret.append(example)

        return ret