from nlplingo.tasks.common.datapoint import Datapoint

class UnaryDatapoint(Datapoint):

    def __init__(self, span, event_domain, embedding_vector_size, label_str):
        super(UnaryDatapoint, self).__init__()
        self.span = span
        self.event_domain = event_domain
        self.embedding_vector_size = embedding_vector_size
        self.label_str = label_str
        self.score = 0.0
        self.label = None