# ==== Start of helper classes to be used by TriggerExtractorResultCollection ====
class DocumentPrediction(object):
    def __init__(self, docid):
        self.docid = docid
        self.sentences = dict()
        """:type: dict[str, SentencePrediction]"""

    def to_json(self):
        d = dict()
        d['docid'] = self.docid
        sentences = []
        for sentence in self.sentences.values():
            sentences.append(sentence.to_json())
        d['sentences'] = sentences
        return d


class SentencePrediction(object):
    def __init__(self, start, end):
        self.id = None
        self.text = None
        self.start = start      # start offset
        self.end = end          # end offset
        self.events = dict()
        """:type: dict[str, EventPrediction]"""
        self.event_event_relations = dict()
        """:type: dict[tuple, SentencePrediction]"""

    def to_json(self):
        d = dict()
        d['start'] = self.start
        d['end'] = self.end
        events = []
        for event in self.events.values():
            events.append(event.to_json())
        d['events'] = events
        event_event_relations = list()
        for event_event_relation in self.event_event_relations.values():
            event_event_relations.append(event_event_relation.to_json())
        d['event_event_relations'] = event_event_relations
        return d


class EventPrediction(object):
    def __init__(self, trigger):
        self.id = None
        self.trigger = trigger  # each EventPrediction is based on a single TriggerPrediction
        """:type: TriggerPrediction"""
        self.arguments = dict()     # but can have multiple arguments
        """:type: dict[str, ArgumentPrediction]"""

    def to_json(self):
        d = dict()
        d['trigger'] = self.trigger.to_json()
        args = []
        for arg in self.arguments.values():
            args.append(arg.to_json())
        d['arguments'] = args
        return d


class TriggerPrediction(object):
    def __init__(self, start, end):
        self.id = None
        self.text = None
        self.start = start
        self.end = end
        self.labels = dict()        # we assume that a single span (start, end) might have multiple predictions
        """:type: dict[str, float]"""

    def to_json(self):
        d = dict()
        d['start'] = self.start
        d['end'] = self.end

        l = dict()
        for label in self.labels:
            l[label] = '%.4f' % (self.labels[label])
        d['labels'] = l
        return d


class ArgumentPrediction(object):
    def __init__(self, start, end):
        self.id = None
        self.text = None
        self.start = start
        self.end = end
        self.labels = dict()
        """:type: dict[str, float]"""
        self.em_id = None       # an argument is usually based on an EntityMention, so we allow capturing of the EntityMention id here

    def to_json(self):
        d = dict()
        d['start'] = self.start
        d['end'] = self.end
        d['em_id'] = self.em_id

        l = dict()
        for label in self.labels:
            l[label] = '%.4f' % (self.labels[label])
        d['labels'] = l
        return d

class EventEventRelationPrediction(object):
    def __init__(self,left_event:EventPrediction,right_event:EventPrediction):
        self.left_event = left_event
        self.right_event = right_event
        self.labels = dict()
        """:type: dict[str, float]"""

    def to_json(self):
        d = dict()
        d['left_event'] = self.left_event.to_json()
        d['right_event'] = self.right_event.to_json()
        for label in self.labels:
            d[label] = '%.4f' % (self.labels[label])
        d['labels'] = d
        return d

# ==== End of helper classes ====
