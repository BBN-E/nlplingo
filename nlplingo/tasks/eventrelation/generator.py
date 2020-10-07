from collections import defaultdict

from nlplingo.text.text_theory import Sentence, annotate_sentence_with_word_embeddings
from nlplingo.text.text_span import Anchor, Token
from nlplingo.common.utils import IntPair, split_offsets

from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.tasks.eventrelation.example import EventEventRelationExample
from nlplingo.text.text_theory import Document as lingoDoc
from nlplingo.text.text_theory import Event
from nlplingo.text.text_span import Anchor
from nlplingo.tasks.common.unary.event_within_sentence import EventWithinSentence

import logging
logger = logging.getLogger(__name__)

class EventEventRelationExampleGenerator(ExampleGenerator):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class to generate candidate Datapoint objects for the event relation
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EventEventRelationExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)

    def generate(self, docs):
        """
        The current event relation generation scheme assumes both events are in the same sentence.
        :param docs:
        :param gen_opennre_triplets:
        :return:
        """
        self.statistics.clear()

        examples = []

        candidate_eer_count = 0

        embedding_vector_size = None
        if 'embeddings' in self.extractor_params:
            if 'vector_size' in self.extractor_params['embeddings']:
                embedding_vector_size = self.extractor_params['embeddings']['vector_size']

        for doc in docs:
            for eer in doc.event_event_relations:
                eer_arg1=eer.arg1
                eer_arg2=eer.arg2
                relation_type=eer.type

                # TODO: this is not efficient
                sentence=None
                for sent in doc.sentences:
                    if eer_arg1.overlaps_with_anchor(sent) and eer_arg2.overlaps_with_anchor(sent):
                        sentence=sent
                if not sentence:
                    print("WARNING: skip EER that is not covered by a single sentence.")
                    continue

                anchor1 = eer_arg1.anchors[0]
                anchor2 = eer_arg2.anchors[0]

                e1 = EventWithinSentence(anchor1, self.event_domain, embedding_vector_size, None, sentence)
                e2 = EventWithinSentence(anchor2, self.event_domain, embedding_vector_size, None, sentence)
                example = EventEventRelationExample(e1, e2, self.event_domain, relation_type, serif_sentence=eer.serif_sentence, serif_event_0=eer.serif_event_0, serif_event_1=eer.serif_event_1)

                examples.append(example)
                candidate_eer_count += 1

        logging.info('candidate_eer_count: %s', candidate_eer_count)
        for k, v in self.statistics.items():
            print('EventEventRelationExampleGenerator stats, {}:{}'.format(k, v))

        return examples

    # Tokenize sentence using whitespace for now
    # Create a nlplingo (Sentence, Anchor1, Anchor2) triplet
    def process_json_ins(self, ins, embeddings, case_sensitive=False):
        if case_sensitive:
            sentence = ' '.join(ins['sentence'].split())
            head = ins['head']['word']
            tail = ins['tail']['word']
        else:
            sentence = ' '.join(ins['sentence'].lower().split())  # delete extra spaces
            head = ins['head']['word'].lower()
            tail = ins['tail']['word'].lower()

        p1 = sentence.find(' ' + head + ' ')
        p2 = sentence.find(' ' + tail + ' ')
        if p1 == -1:
            if sentence[:len(head) + 1] == head + " ":
                p1 = 0
            elif sentence[-len(head) - 1:] == " " + head:
                p1 = len(sentence) - len(head)
            else:
                p1 = 0  # shouldn't happen
                raise Exception('entity not found')
        else:
            p1 += 1
        if p2 == -1:
            if sentence[:len(tail) + 1] == tail + " ":
                p2 = 0
            elif sentence[-len(tail) - 1:] == " " + tail:
                p2 = len(sentence) - len(tail)
            else:
                p2 = 0  # shouldn't happen
        else:
            p2 += 1
        if p1 == -1 or p2 == -1:
             raise Exception("[ERROR] Sentence doesn't contain the entity, sentence = {}, head = {}, tail = {}".format(sentence, head, tail))

        token_lst = []
        offsets = split_offsets(sentence)
        anchor1_token_index = -1
        anchor2_token_index = -1
        for i, offsets in enumerate(offsets):
            token_lst.append(Token(IntPair(None, None), i, offsets[0], None, None))
            if offsets[1] <= p1 <= offsets[2]:
                anchor1_token_index = i

            if offsets[1] <= p2 <= offsets[2]:
                anchor2_token_index= i

        assert(anchor1_token_index != -1)
        assert(anchor2_token_index != -1)
        sentence = Sentence(None, IntPair(None, None), None, token_lst, None, add_noun_phrases=False)
        annotate_sentence_with_word_embeddings(sentence, embeddings)
        anchor1 = Anchor(None, IntPair(None, None), None, None)
        tokens1 = [sentence.tokens[anchor1_token_index]]
        anchor1.with_tokens(tokens1)
        anchor2 = Anchor(None, IntPair(None, None), None, None)
        tokens2 = [sentence.tokens[anchor2_token_index]]
        anchor2.with_tokens(tokens2)
        return (sentence, anchor1, anchor2)


    def generate_json(self, json_objs, feature_generator, embeddings):
        """
        Generate features for a set of json_objs.
        :param json_objs: list of JSON entries
        :param feature_generator: nlplingo.tasks.common.generator.FeatureGenerator
        :return:
        """
        self.statistics.clear()

        examples = []

        total_eer_count = 0
        eer_count = 0

        for json_obj in json_objs:
            # directly construct sentence, anchor1, anchor2 in nlplingo
            sentence, anchor1, anchor2 = self.process_json_ins(json_obj, embeddings)
            relation_type = json_obj['relation']

            example = EventEventRelationExample(anchor1, anchor2, sentence, self.event_domain,
                                           self.extractor_params, feature_generator.features, relation_type)
            feature_generator.generate_example(example, sentence.tokens, self.hyper_params)
            examples.append(example)
            eer_count += 1
            total_eer_count += 1

        print('total_eer_count', total_eer_count)
        print('eer_count', eer_count)
        for k, v in self.statistics.items():
            print('EventEventRelationExampleGenerator stats, {}:{}'.format(k, v))

        return examples
