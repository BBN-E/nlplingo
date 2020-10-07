
from nlplingo.tasks.common.sequence.base import SequenceDatapoint


class SequenceExample(SequenceDatapoint):
    # +1
    def __init__(self, sentence_text, words, labels, sentence, event_domain):
        """
        :type sentence_text: str
        :type words: list[str]
        :type labels: list[str]
        :type sentence: nlplingo.text.text_span.Sentence
        """
        super(SequenceExample, self).__init__(sentence_text, words, labels)

        self.sentence = sentence
        self.docid = None           # docid and sentence_index are used for decoding
        self.sentence_index = None  # so that we know which sentence this example corresponds to
        self.event_domain = event_domain
        self._allocate_arrays()

    @property
    def sentence_text(self):
        """:rtype: str"""
        return self.text

    @sentence_text.setter
    def sentence_text(self, text):
        """:type text: str"""
        self.text = text

    @property
    def words(self):
        """:rtype: list[str]"""
        return self.elements

    @words.setter
    def words(self, words):
        """:type text: list[str]"""
        self.elements = words

    @property
    def labels(self):
        """:rtype: list[str]"""
        return self.label_strings

    @labels.setter
    def labels(self, labels):
        """:type text: list[str]"""
        self.label_strings = labels

    # +1
    def _allocate_arrays(self):
        """
        This is more like specifying the different feature variables.
        But to be consistent with the rest of NLPLingo, we will still name the method as _allocate_arrays
        :return:
        """
        self.input_ids = None
        self.input_mask = None
        self.segment_ids = None
        self.label_ids = None
        self.subword_to_token_indices = None
        self.tokens = None
        self.seq_length = None
