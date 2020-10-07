import random
from collections import defaultdict

from nlplingo.tasks.entitycoref.example import EntityCorefExample
from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.tasks.common.unary.entity_within_sentence import EntityWithinSentence

import logging
logger = logging.getLogger(__name__)


class EntityCorefExampleGenerator(ExampleGenerator):
    def __init__(self, domain, params, extractor_params, hyper_params):
        super(EntityCorefExampleGenerator, self).__init__(domain, params, extractor_params, hyper_params)
        if hasattr(self.hyper_params, "num_batches"):
            self.max_positive_examples = int(extractor_params['max_positive_examples']) // self.hyper_params.num_batches
            self.max_negative_examples = int(extractor_params['max_negative_examples']) // self.hyper_params.num_batches
        else:
            self.max_positive_examples = int(extractor_params['max_positive_examples'])
            self.max_negative_examples = int(extractor_params['max_negative_examples'])
        logging.info("max_positive_examples: %s", extractor_params['max_positive_examples'])
        self.statistics = defaultdict(int)

    # TODO there needs to be a sampling during training, else N-choose-2 will generate too many training pairs (especially negatives)
    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type feature_generator: nlplingo.tasks.entitycoref.feature.EntityCorefFeatureGenerator
        :type mode: str     # either: train, dev, test
        :rtype: list[nlplingo.tasks.entitycoref.example.EntityCorefExample]
        """
        ret = []
        self.statistics.clear()

        positive_examples = []
        negative_examples = []

        embedding_vector_size = self.extractor_params['embeddings']['vector_size']

        for doc in docs:
            # first, gather the mappings from entity_mention_id to entity_id
            mention_to_entity_id = dict()
            for entity in doc.entities_by_id.values():
                for mention in entity.mentions:
                    mention_to_entity_id[mention.id] = entity.id

            # now collect all entity mentions in doc into a list
            mentions = []
            """:type: list[nlplingo.text.text_span.EntityMention]"""
            mention_id_to_sentence_index = dict()
            for i, sentence in enumerate(doc.sentences):
                for mention in sentence.entity_mentions:
                    mentions.append(mention)
                    mention_id_to_sentence_index[mention.id] = i

            # now generate N-choose-2 pairs
            for i in range(len(mentions)):
                m1 = mentions[i]
                m1_sentence = doc.sentences[mention_id_to_sentence_index[m1.id]]
                eid1 = mention_to_entity_id[m1.id]
                for j in range(i + 1, len(mentions)):
                    m2 = mentions[j]
                    m2_sentence = doc.sentences[mention_id_to_sentence_index[m2.id]]
                    eid2 = mention_to_entity_id[m2.id]

                    label = int(eid1 == eid2)

                    e1 = EntityWithinSentence(m1, self.event_domain, embedding_vector_size, None, m1_sentence)
                    e2 = EntityWithinSentence(m2, self.event_domain, embedding_vector_size, None, m2_sentence)

                    example = EntityCorefExample(e1, e2, self.event_domain, label)

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
            print('EntityCorefExampleGenerator stats, {}:{}'.format(k, v))

        return ret
