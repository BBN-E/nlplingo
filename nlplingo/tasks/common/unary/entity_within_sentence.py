from nlplingo.common.utils import Struct
from nlplingo.tasks.common.unary.within_sentence import UnaryWithinSentence
import numpy as np

class EntityWithinSentence(UnaryWithinSentence):

    def __init__(self, span, event_domain, embedding_vector_size,
                 label_str, sentence, head_only=False):
        super(EntityWithinSentence, self).__init__(span, event_domain, embedding_vector_size, label_str, sentence)

        if not head_only:
            self.span_token_indices = Struct(start=self.span.tokens[0].index_in_sentence,
                                          end=self.span.tokens[-1].index_in_sentence, head=self.span.head().index_in_sentence)
        else:
            head_token_index = span.head().index_in_sentence
            self.span_token_indices = Struct(start=head_token_index, end=head_token_index, head=head_token_index)