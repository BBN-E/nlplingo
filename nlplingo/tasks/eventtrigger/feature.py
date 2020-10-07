from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy as np

from nlplingo.tasks.common.unary.event_within_sentence import EventWithinSentenceFeatureGenerator

class EventTriggerFeatureGenerator(EventWithinSentenceFeatureGenerator):
    # we only accept tokens of the following part-of-speech categories as trigger candidates
    # trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
    trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])

    def __init__(self, extractor_params, hyper_params, feature_setting, domain):
        super(EventTriggerFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)
        self.mask_implicit_negatives = extractor_params.get('mask_implicit_negatives', False)
        self.mask_implicit_negatives_value = -1
        self.is_hierarchical = extractor_params.get('hierarchical')
        self.label_hierarchy = self._label_hierarchy_builder(domain)

    def populate_example(self, example):
        """
        :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        super(EventTriggerFeatureGenerator, self).populate_example(example)

        event_domain = example.event_domain

        if isinstance(example.event_type, str):
            event_type_indices = [event_domain.get_event_type_index(example.event_type)]
        else:  # a collection (multilabel)
            event_type_indices = [event_domain.get_event_type_index(t) for t in example.event_type]

        if self.mask_implicit_negatives:
            # currently, mask ALL NEGATIVES
            for idx in range(example.label.shape[0]):
                example.label[idx] = self.mask_implicit_negatives_value

        for idx in event_type_indices:
            example.label[idx] = 1

        if self.is_hierarchical:  # relabel hierarchically
            new_labels = np.asarray([example.label])
            self.label_hierarchy(new_labels)
            example.label = new_labels[0]

    def _label_hierarchy_builder(self, domain):

        def labeler(labels_array):
            # for *each* True label l, set *all* of l's ancestor labels to True.
            # Positives added here will override masked (implicit) negatives.
            prior = domain.hierarchical_prior
            for candidate_index, sample in enumerate(labels_array):
                implicit = sample == self.mask_implicit_negatives_value
                true_labels = sample == 1
                for label_index, label in enumerate(sample):
                    true_ancestors = label * prior[label_index] == 1
                    sample = np.logical_or(true_labels, true_ancestors)
                # keep implicit negatives that are not ancestors of true labels
                overridden = np.logical_and(implicit, sample)
                still_implicitly_negative = implicit > overridden
                new_sample = np.where(still_implicitly_negative,
                                      self.mask_implicit_negatives_value,
                                      sample)
                labels_array[candidate_index] = new_sample

        return labeler

    @staticmethod
    def get_event_type_of_token(token, sent):
        """:type token: nlplingo.text.text_span.Token"""
        event_type = 'None'
        # print('target token, ', token.to_string())
        for event in sent.events:
            for anchor in event.anchors:
                # print('checking against anchor, ', anchor.to_string())
                if token.start_char_offset() == anchor.head().start_char_offset() and token.end_char_offset() == anchor.head().end_char_offset():
                    event_type = event.label
                    break
        return event_type

    @staticmethod
    def get_event_types_of_token(token, sent):
        """:type token: nlplingo.text.text_span.Token"""
        #event_types = {'positive': [], 'negative': []}
        event_types = set(['None'])
        # annotation produces separate doc.events and sent.events for each spanfile
        for event in sent.events:
            for anchor in event.anchors:
                # print('checking against anchor, ', anchor.to_string())
                if token.start_char_offset() == anchor.head().start_char_offset() and token.end_char_offset() == anchor.head().end_char_offset():
                    event_type = event.label
                    event_types.add(event_type)
                    #if event_type.startswith('-'):
                    #    event_types['negative'].append(event_type.strip('-'))
                    #else:
                    #    event_types['positive'].append(event_type)
        if 'None' in event_types and len(event_types) > 1:
            event_types.remove('None')
        return event_types

    @staticmethod
    def get_event_type_of_np(np, sent):
        """:type token: nlplingo.text.text_span.TextSpan"""
        event_type = 'None'
        for event in sent.events:
            for anchor in event.anchors:
                if anchor.start_char_offset() == np.start_char_offset() and anchor.end_char_offset() == np.end_char_offset():
                    event_type = event.label
                    break
        return event_type