import sys
import random
from collections import defaultdict

from nlplingo.tasks.eventpair.example import EventPairExample
from nlplingo.tasks.common.examplegenerator import ExampleGenerator


class EventPairExampleGenerator(ExampleGenerator):
    # +1
    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        A class to generate candidate Datapoint objects for the event coref
        task.  Does not populate features.  The generate method can also conduct
        sampling and similar preprocessing.
        :param event_domain: EventDomain object to get label data
        :param params: run-level config containing generation parameters
        :param extractor_params: extractor-level config with generation params
        :param hyper_params: extractor-level hyperparameters used in generation
        """
        super(EventPairExampleGenerator, self).__init__(event_domain, params, extractor_params, hyper_params)

        self.max_train_samples_per_class = extractor_params.get('max_train_samples_per_class')
        self.max_dev_samples_per_class = extractor_params.get('max_dev_samples_per_class')
        self.max_test_samples_per_class = extractor_params.get('max_test_samples_per_class')
        self.num_test_queries_per_class = extractor_params.get('max_train_samples_per_class')
        self.max_test_none_samples = extractor_params.get('max_test_none_samples')
        if self.num_test_queries_per_class < self.max_test_samples_per_class:
            print("Warning: num_test_queries_per_class = max_train_samples_per_class < max_test_samples_per_class")

        self.max_pair_samples = extractor_params.get('max_pair_samples', sys.maxsize)
        self.max_test_samples = extractor_params.get('max_test_samples', sys.maxsize)
        #self.max_train_samples = extractor_params['max_train_samples']

        self.information = []


    def get_id(self, eg):
        """
        :type eg: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        """
        s = '%s_%d-%d' % (eg.sentence.docid, eg.anchor.start_char_offset(), eg.anchor.end_char_offset())
        return s


    # +1
    # currently using the following function to generate training examples.
    # e.g. given 3 event types or classes: A, B, C ; we generate training examples as follows
    # 'SAME': AxA, BxB, CxC
    # 'DIFFERENT': AxB, AxC, BxC
    # In each of the above cross-products or pairings, we restrict to self.max_pair_samples
    # e.g. AxA, AxB, etc. each of these must generate <= self.max_pair_samples candidates. We randomly sample self.max_pair_samples
    def generate_train(self, examples, max_samples_per_class):
        """
        :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        :rtype: list[nlplingo.tasks.eventpair.example.EventPairExample]
        """
        self.statistics.clear()
        number_positive = 0
        number_negative = 0
    
        ret = []
        """:type: list[nlplingo.tasks.eventpair.example.EventPairExample]"""
    
        # first, let's group examples by event_type
        example_by_event_type = defaultdict(list)
        """:type: dict[str, list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]]"""

        random.shuffle(examples)

        for example in examples:
            if len(example_by_event_type[example.event_type]) < max_samples_per_class:
                example_by_event_type[example.event_type].append(example)

        event_types = [et for et in sorted(example_by_event_type.keys())]

        self.event_definition_examples = defaultdict(list)
        print('#### In generate_train, example_by_event_type')
        for et in sorted(example_by_event_type):
            print('#%s = %d' % (et, len(example_by_event_type[et])))
            for eg in example_by_event_type[et]:
                self.information.append('TRAIN %s %s' % (et, self.get_id(eg)))
                self.event_definition_examples[et].append(eg)
        print('')

        # now let's pair up the examples. Intra-class are positives, Inter-class are negatives
        for label_index, label in enumerate(event_types):
            label_examples = example_by_event_type.get(label)
            self.statistics['#{}'.format(label)] = len(label_examples)
    
            # first, the positive examples: intra-class for all positive classes
            candidates = []
            if label != 'None': # 'None' examples should not be paired with other 'None' to form positive pairs
                for i in range(len(label_examples) - 1):
                    for j in range(i + 1, len(label_examples)):
                        candidates.append(EventPairExample(label_examples[i], label_examples[j], 'SAME'))
                        self.statistics['P #{}'.format(label)] += 1

            # generating all pairs of trigger examples will be too much. So we place limits on each intra-class pair generation
            if len(candidates) > self.max_pair_samples:
                selected_candidates = random.sample(candidates, self.max_pair_samples)
            else:
                selected_candidates = candidates
            number_positive += len(selected_candidates)
            ret.extend(selected_candidates)
            self.statistics['S-P #{}'.format(label)] += len(selected_candidates)
    
            # now, the negative examples: inter-class over pairs of positive classes
            for other_index in range(label_index+1, len(event_types)):
                other_label = event_types[other_index]
    
                other_examples = example_by_event_type.get(other_label)
                candidates = []
                for eg1 in label_examples:
                    for eg2 in other_examples:
                        candidates.append(EventPairExample(eg1, eg2, 'DIFFERENT'))
                        self.statistics['N #{}_{}'.format(label, other_label)] += 1

                # generating all pairs of trigger examples will be too much. So we place limits on each inter-class pair generation
                if len(candidates) > self.max_pair_samples:
                    selected_candidates = random.sample(candidates, self.max_pair_samples)
                else:
                    selected_candidates = candidates
                number_negative += len(selected_candidates)
                ret.extend(selected_candidates)
                self.statistics['S-N #{}_{}'.format(label, other_label)] += len(selected_candidates)
    
        for k in sorted(self.statistics.keys()):
            print('EventPairGenerator stats, {}:{}'.format(k, self.statistics.get(k)))
        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))
        return ret

    # The following is not used. This is potentially another way to generate training examples, which follows the way of generating test examples.
    # Note the above 'generate_train' function will generate training examples that is different in distribution with generation of test examples.
    # So an alternative approach (below), is to generate all pairs first: AxA, BxB, CxC, AxB, AxC, BxC , and then randomly sample N number of examples
    # def generate_train2(self, examples):
    #     """
    #     :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]
    #     :rtype: list[nlplingo.tasks.eventpair.example.EventPairExample]
    #     """
    #     number_positive = 0
    #     number_negative = 0
    #
    #     ret = []
    #     """:type: list[nlplingo.tasks.eventpair.example.EventPairExample]"""
    #
    #     for i, example in enumerate(examples):
    #         eg1 = examples[i]
    #         for j in range(i+1, len(examples)):
    #             eg2 = examples[j]
    #
    #             if eg1.event_type == eg2.event_type:
    #                 if eg1.event_type != 'None':    # since given two 'None'-'None', we cannot say whether they are same or different
    #                     ret.append(EventPairExample(eg1, eg2, 'SAME'))
    #                     number_positive += 1
    #             else:
    #                 ret.append(EventPairExample(eg1, eg2, 'DIFFERENT'))
    #                 number_negative += 1
    #
    #     print('#Positives = {}'.format(number_positive))
    #     print('#Negatives = {}'.format(number_negative))
    #
    #     random.shuffle(ret)
    #
    #     print('Randomly sampling {} train samples'.format(self.max_train_samples))
    #     return ret[0:self.max_train_samples]

    # +1
    # we simply generate all pairs: AxA, BxB, CxC, AxB, AxC, BxC
    def generate_dev(self, examples, max_samples_per_class):
        """
        :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        :rtype: list[nlplingo.tasks.eventpair.example.EventPairExample]
        """
        number_positive = 0
        number_negative = 0

        ret = []
        """:type: list[nlplingo.tasks.eventpair.example.EventPairExample]"""

        example_by_event_type = defaultdict(int)    # keep track of how many examples we've added per event type

        random.shuffle(examples)
        filtered_examples = []
        """type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""
        for example in examples:
            if example_by_event_type[example.event_type] < max_samples_per_class:
                example_by_event_type[example.event_type] += 1
                filtered_examples.append(example)

        et_counter = defaultdict(int)
        for eg in filtered_examples:
            et_counter[eg.event_type] += 1
        print('#### In generate_dev, filtered_examples')
        for et in sorted(et_counter):
            print('#%s = %d' % (et, et_counter[et]))
        print('')
        
        for eg in filtered_examples:
            self.information.append('DEV %s %s' % (eg.event_type, self.get_id(eg)))

        for i in range(len(filtered_examples)-1):
            eg1 = filtered_examples[i]
            for j in range(i+1, len(filtered_examples)):
                eg2 = filtered_examples[j]

                if eg1.event_type == eg2.event_type:
                    if eg1.event_type != 'None':    # since given two 'None'-'None', we cannot say whether they are same or different
                        ret.append(EventPairExample(eg1, eg2, 'SAME'))
                        number_positive += 1
                else:
                    ret.append(EventPairExample(eg1, eg2, 'DIFFERENT'))
                    number_negative += 1

        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))

        random.shuffle(ret)

        print('Randomly sampling {} test samples'.format(self.max_test_samples))
        return ret[0:self.max_test_samples]


    def generate_test(self, examples, max_samples_per_class):
        """
        :type examples: list[nlplingo.tasks.eventtrigger.example.EventTriggerExample]
        :rtype: list[nlplingo.tasks.eventpair.example.EventPairExample]
        """
        number_of_queries = self.num_test_queries_per_class

        number_positive = 0
        number_negative = 0

        ret = []
        """:type: list[nlplingo.tasks.eventpair.example.EventPairExample]"""

        # first, let's group examples by event_type
        example_by_event_type = defaultdict(list)
        """:type: dict[str, list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]]"""

        random.shuffle(examples)

        for example in examples:
            if example.event_type == 'None':
                if len(example_by_event_type[example.event_type]) < self.max_test_none_samples:
                    example_by_event_type[example.event_type].append(example)
            else:
                if len(example_by_event_type[example.event_type]) < max_samples_per_class:
                    example_by_event_type[example.event_type].append(example)

        query_examples = []
        unlabeled_examples = []
        for et in example_by_event_type:
            egs = example_by_event_type[et]

            # we use the first 'number_of_queries' as the queries
            if et != 'None':
                #query_examples.extend(egs[0:number_of_queries])
                query_examples.extend(self.event_definition_examples[et])
            #unlabeled_examples.extend(egs[number_of_queries:])
            unlabeled_examples.extend(egs)

        et_counter = defaultdict(int)
        for eg in query_examples:
            et_counter[eg.event_type] += 1
        print('#### In generate_test, query_examples')
        for et in sorted(et_counter):
            print('#%s = %d' % (et, et_counter[et]))
        print('')
        et_counter = defaultdict(int)
        for eg in unlabeled_examples:
            et_counter[eg.event_type] += 1
        print('#### In generate_test, unlabeled_examples')
        for et in sorted(et_counter):
            print('#%s = %d' % (et, et_counter[et]))
        print('')

        for eg in query_examples:
            self.information.append('QUERY %s %s' % (eg.event_type, self.get_id(eg)))
        for eg in unlabeled_examples:
            self.information.append('TEST %s %s' % (eg.event_type, self.get_id(eg)))

        for eg1 in query_examples:
            for eg2 in unlabeled_examples:
                if eg1.event_type == eg2.event_type:
                    if eg1.event_type != 'None':  # since given two 'None'-'None', we cannot say whether they are same or different
                        ret.append(EventPairExample(eg1, eg2, 'SAME'))
                        number_positive += 1
                else:
                    ret.append(EventPairExample(eg1, eg2, 'DIFFERENT'))
                    number_negative += 1

        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))

        random.shuffle(ret)

        print('Randomly sampling {} test samples'.format(self.max_test_samples))
        return ret[0:self.max_test_samples]



