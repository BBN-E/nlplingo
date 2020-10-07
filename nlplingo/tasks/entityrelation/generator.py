from nlplingo.tasks.common.unary.entity_within_sentence import EntityWithinSentence
from nlplingo.text.text_span import spans_overlap
from nlplingo.tasks.common.examplegenerator import ExampleGenerator
from nlplingo.tasks.entityrelation.example import EntityRelationExample

class EntityRelationExampleGenerator(ExampleGenerator):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class for generating candidate Datapoint objects for the relation
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EntityRelationExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)


    def generate(self, docs):
        """
        The current entity relation generation scheme assumes both entities are in the same sentence.
        :param docs:
        :return:
        """
        self.statistics.clear()

        examples = []
        embedding_vector_size = self.extractor_params['embeddings']['vector_size']

        for doc in docs:
            for eer in doc.entity_entity_relations:
                eer_arg1=eer.arg1
                eer_arg2=eer.arg2
                relation_type=eer.type

                # TODO: this is not efficient
                sentence=None
                for sent in doc.sentences:
                    if spans_overlap(eer_arg1, sent) and spans_overlap(eer_arg2, sent):
                        sentence=sent
                if not sentence:
                    print("WARNING: skip entity-relation that is not covered by a single sentence.")
                    continue

                e1 = EntityWithinSentence(eer_arg1, self.event_domain, embedding_vector_size, None, sentence, head_only=True)
                e2 = EntityWithinSentence(eer_arg2, self.event_domain, embedding_vector_size, None, sentence, head_only=True)

                example = EntityRelationExample(e1, e2, self.event_domain, relation_type)
                examples.append(example)

        for k, v in self.statistics.items():
            print('EntityRelationExampleGenerator stats, {}:{}'.format(k, v))

        return examples


