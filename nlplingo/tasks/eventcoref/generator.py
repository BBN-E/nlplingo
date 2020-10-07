import random
from collections import defaultdict

from nlplingo.tasks.eventcoref.example import EventCorefExample
from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.tasks.common.unary.event_within_sentence import EventWithinSentence

class EventCorefExampleGenerator(ExampleGenerator):
    def __init__(self, domain, params, extractor_params, hyper_params):
        super(EventCorefExampleGenerator, self).__init__(domain, params, extractor_params, hyper_params)
        if hasattr(self.hyper_params, "num_batches"):
            self.max_positive_examples = int(extractor_params['max_positive_examples']) // self.hyper_params.num_batches
            self.max_negative_examples = int(extractor_params['max_negative_examples']) // self.hyper_params.num_batches
        else:
            self.max_positive_examples = int(extractor_params['max_positive_examples'])
            self.max_negative_examples = int(extractor_params['max_negative_examples'])
        self.statistics = defaultdict(int)

    # TODO there needs to be a sampling during training, else N-choose-2 will generate too many training pairs (especially negatives)
    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :rtype: list[nlplingo.tasks.eventcoref.example.EventCorefExample]
        """
        ret = []
        self.statistics.clear()

        positive_examples = []
        negative_examples = []

        embedding_vector_size = self.extractor_params['embeddings']['vector_size']

        for doc in docs:
            # first, generate the mapping from event_id to coref_id
            anchor_to_coref_id = dict()
            for i, coref_event in enumerate(doc.coref_events):  # for each set of coref events
                for event in coref_event:                       # for each event in the set
                    anchor = event.anchors[0]
                    anchor_to_coref_id[anchor.id] = i

            # now collect all anchors in doc into a list
            anchors = []
            """:type: list[nlplingo.text.text_span.Anchor]"""
            anchor_id_to_sentence_index = dict()
            for i, sentence in enumerate(doc.sentences):
                for event in sentence.events:
                    anchor = event.anchors[0]
                    anchors.append(anchor)            # TODO this currently assumes all events have just a single anchor
                    anchor_id_to_sentence_index[anchor.id] = i

            # now generate N-choose-2 pairs
            for i in range(len(anchors)):
                m1 = anchors[i]
                m1_sentence = doc.sentences[anchor_id_to_sentence_index[m1.id]]
                eid1 = anchor_to_coref_id[m1.id]
                for j in range(i + 1, len(anchors)):
                    m2 = anchors[j]
                    m2_sentence = doc.sentences[anchor_id_to_sentence_index[m2.id]]
                    eid2 = anchor_to_coref_id[m2.id]

                    label = int(eid1 == eid2)

                    e1 = EventWithinSentence(m1, self.event_domain, embedding_vector_size, None, m1_sentence)
                    e2 = EventWithinSentence(m2, self.event_domain, embedding_vector_size, None, m2_sentence)
                    example = EventCorefExample(e1, e2, self.event_domain, label)

                    if self.train_dev_test_mode == 'train':     # do sampling
                        if label:   # positive
                            if len(positive_examples) < self.max_positive_examples:
                                positive_examples.append(example)
                            else:
                                index = random.randint(0, self.max_positive_examples-1)
                                positive_examples[index] = example
                        else:       # negative
                            if len(negative_examples) < self.max_negative_examples:
                                negative_examples.append(example)
                            else:
                                index = random.randint(0, self.max_negative_examples-1)
                                negative_examples[index] = example
                    else:
                        if label:
                            self.statistics['num# positive'] += 1
                        else:
                            self.statistics['num# negative'] += 1
                        ret.append(example)

        if self.train_dev_test_mode == 'train':
            print('num# positive=', len(positive_examples))
            print('num# negative=', len(negative_examples))
            ret.extend(positive_examples)
            ret.extend(negative_examples)
            random.shuffle(ret)

        for k, v in self.statistics.items():
            print('EventCorefExampleGenerator stats, {}:{}'.format(k, v))

        return ret