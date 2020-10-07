from nlplingo.tasks.common.datapoint import Datapoint

class BinaryDatapoint(Datapoint):

    def __init__(self, arg0, arg1, event_domain, label_str):
        """
        :type arg0: UnaryDatapoint
        :type arg1: UnaryDatapoint
        """
        super(BinaryDatapoint, self).__init__()
        self.arg0 = arg0
        self.arg1 = arg1
        self.event_domain = event_domain
        self.label_str = label_str
        self.score = 0.0
        self.label = None
