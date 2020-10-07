import codecs
import random
from collections import defaultdict

from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator
from nlplingo.tasks.eventtrigger.example import EventTriggerExample
from nlplingo.tasks.eventargument.example import EventArgumentExample
from nlplingo.tasks.eventframe.example import EventFrameExample
from nlplingo.tasks.eventframe.example import EventFramePairExample
from nlplingo.tasks.common.examplegenerator import ExampleGenerator


class EventFramePairExampleGenerator(ExampleGenerator):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class to generate candidate Datapoint objects for the event frame pair
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EventFramePairExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)


    def generate(self, docs, trigger_feature_generator, argument_feature_generator):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type trigger_feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
        :type argument_feature_generator: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator
        """
        self.statistics.clear()

        frame_examples = defaultdict(list)  # docid -> list[EventFrameExample]
        """:type: dict[str, list[nlplingo.tasks.eventframe.example.EventFrameExample]]"""
        for doc in docs:
            for sent in doc.sentences:
                frame_examples[doc.docid].extend(self._generate_sentence_frames(sent, trigger_feature_generator, argument_feature_generator))

        framepair_examples = self._generate_framepairs(frame_examples)
        """:type: list[nlplingo.tasks.eventframe.example.EventFramePairExample]"""

        for k, v in self.statistics.items():
            print('EventFramePairExampleGenerator stats, {}:{}'.format(k, v))

        return framepair_examples


    def _generate_sentence_frames(self, sentence, trigger_feature_generator, argument_feature_generator):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type trigger_feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
        :type argument_feature_generator: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator
        """
        embedding_vector_size = self.extractor_params['embeddings']['vector_size']

        ret = []
        """:type: list[nlplingo.tasks.eventframe.example.EventFrameExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.hyper_params.max_sentence_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        for event in sentence.events:
            for anchor in event.anchors:
                trigger_eg = EventTriggerExample(anchor, sentence, self.event_domain, embedding_vector_size, anchor.label)
                trigger_feature_generator.generate_example(trigger_eg, self.hyper_params)

                agents = [arg for arg in event.arguments if arg.label.lower() == 'agent']
                patients = [arg for arg in event.arguments if arg.label.lower() == 'patient']

                for agent in agents:
                    agent_eg = EventArgumentExample(anchor, agent.entity_mention, sentence, self.event_domain,
                                                    self.extractor_params, argument_feature_generator.features,
                                                    self.hyper_params, agent.label)
                    argument_feature_generator.generate_example(agent_eg, sentence.tokens, self.hyper_params)

                    for patient in patients:
                        patient_eg = EventArgumentExample(anchor, patient.entity_mention, sentence, self.event_domain,
                                                          self.extractor_params, argument_feature_generator.features,
                                                          self.hyper_params, patient.label)
                        argument_feature_generator.generate_example(patient_eg, sentence.tokens, self.hyper_params)

                        ret.append(EventFrameExample(trigger_eg, [agent_eg, patient_eg], self.extractor_params, self.hyper_params))

        return ret


    def _generate_framepairs(self, frame_examples):
        """

        :param frame_examples: dict[str, list[nlplingo.tasks.eventframe.example.EventFrameExample]]
        :return:
        """
        ret = []
        """:type: list[nlplingo.tasks.eventframe.example.EventFramePairExample]"""

        same_pairs = []
        different_pairs = []

        docids = sorted(frame_examples.keys())

        for i in range(0, len(docids) - 1):
            frames1 = frame_examples[docids[i]]
            """:type: list[nlplingo.tasks.eventframe.example.EventFrameExample]"""

            for j in range(i + 1, len(docids)):
                frames2 = frame_examples[docids[j]]
                """:type: list[nlplingo.tasks.eventframe.example.EventFrameExample]"""

                for f1 in frames1:
                    for f2 in frames2:
                        if f1.trigger_example.event_type == f2.trigger_example.event_type:
                            same_pairs.append(EventFramePairExample(f1, f2, 'SAME'))
                        else:
                            different_pairs.append(EventFramePairExample(f1, f2, 'DIFFERENT'))

        random.shuffle(same_pairs)
        ret.extend(same_pairs[0:int(0.1 * len(same_pairs))])

        random.shuffle(different_pairs)
        ret.extend(different_pairs[0:int(0.1 * len(different_pairs))])

        random.shuffle(ret)

        return ret



