from nlplingo.tasks.common.datapoint import Datapoint

class SequenceDatapoint(Datapoint):

    def __init__(self, text, elements, label_strings):
        super(SequenceDatapoint, self).__init__()
        self.text = text
        self.elements = elements
        self.label_strings = label_strings
        self.score = 0.0
        self.label = None  # a 2d vector
        # 1 to 1 correspondence between words and labels
        assert len(elements) == len(label_strings)
