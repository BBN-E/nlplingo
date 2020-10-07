
import numpy as np

from nlplingo.tasks.common.unary.event_within_sentence import EventWithinSentence
from nlplingo.common.data_types import int_type

class EventTriggerExample(EventWithinSentence):
    def __init__(self, anchor, sentence, event_domain, embedding_vector_size,
                 event_type=None):
        """We are given a token, sentence as context, and event_type (present during training)
        :type anchor: nlplingo.text.text_span.Anchor
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type event_type: str
        """
        super(EventTriggerExample, self).__init__(anchor,
                                                  event_domain,
                                                  embedding_vector_size,
                                                  event_type,
                                                  sentence)

        num_labels = len(self.event_domain.event_types)
        self.label = np.zeros(num_labels, dtype=int_type)

    def __str__(self):
        s = ('<EventTriggerDatapoint type="{}" anchor="{}" sentence="{}" '
             'score={}>'.format(self.event_type, self.anchor.text,
                                self.sentence.text, self.score))
        return s

    def get_keyword(self):
        return self.anchor.head().text.lower()

    def is_na(self):
        if isinstance(self.event_type, str):
            return self.event_type == "None"
        else:
            return "None" in self.event_type or len(self.event_type) == 0
