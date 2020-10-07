import argparse
import json
from itertools import chain

__FORMAT_TYPE__ = "bp-corpus"
__FORMAT_VERSION__ = "v8f"


class Corpus:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf8') as f:
            data = json.load(f)
        self.__corpus_id = data['corpus-id']
        self.__format_type = data['format-type']
        self.__format_version = data['format-version']
        self.__provenance = data['provenance']
        assert(self.__format_type == __FORMAT_TYPE__)
        assert(self.__format_version == __FORMAT_VERSION__)
        self.__docs = dict()
        for entry_id, entry_value in data['entries'].items():
            assert(entry_id == entry_value['entry-id'])
            doc_id = entry_value['doc-id']
            if doc_id not in self.__docs:
                self.__docs[doc_id] = Document(doc_id)
            self.__docs[doc_id].add_entry(entry_value)
        for doc in self.__docs.values():
            doc.sort_sentences()

    @property
    def format_type(self):
        return self.__format_type

    @property
    def format_version(self):
        return self.__format_version

    @property
    def docs(self):
        """
        Returns:
            dict
        """
        return self.__docs

    def clear_annotation(self):
        for doc in self.docs.values():
            doc.clear_annotation()

    def save(self, output_file):
        entries = {}
        for _, doc in self.docs.items():
            doc_entries = doc.to_json_dict()
            for entry_id, entry_value in doc_entries.items():
                assert(entry_id not in entries)
                entries[entry_id] = entry_value
        data = {
            'corpus-id': self.__corpus_id,
            'entries': entries,
            'format-type': self.__format_type,
            'format-version': self.__format_version,
            'provenance': self.__provenance
        }
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(
                data, output, ensure_ascii=False, indent=2, sort_keys=True)


class Document:
    def __init__(self, doc_id):
        self.__doc_id = doc_id
        self.__sentences = []

    @property
    def doc_id(self):
        return self.__doc_id

    @property
    def sentences(self):
        return self.__sentences

    def add_entry(self, entry_dict):
        if entry_dict['segment-type'] == 'sentence':
            sentence = Sentence(doc_id=self.doc_id, entry_dict=entry_dict)
            self.__sentences.append(sentence)
        else:
            raise RuntimeError(
                'segment-type: {} not implemented!'.format(
                    entry_dict['segment-type']
                )
            )

    def sort_sentences(self):
        self.__sentences = sorted(self.__sentences, key=lambda x: x.sent_id)

    @property
    def text(self):
        out_text = ''
        for sentence in self.__sentences:
            out_text += sentence.text + "\n"
        return out_text

    def __repr__(self):
        return self.text

    def clear_annotation(self):
        for sentence in self.sentences:
            sentence.clear_annotation()

    def to_json_dict(self):
        entries = {}
        for sentence in self.sentences:
            entries[sentence.entry_id] = sentence.to_json_dict()
        return entries


class Sentence:
    def __init__(self, *, doc_id, entry_dict):
        assert(entry_dict['segment-type'] == 'sentence')
        self.__abstract_events = dict()
        self.__span_sets = dict()
        self.__doc_id = doc_id
        self.__text = entry_dict['segment-text']
        self.__sent_id = int(entry_dict['sent-id'])
        self.__entry_id = entry_dict['entry-id']

        # Right now we are assuming all entries have abstract-events
        abstract_events_data = entry_dict['annotation-sets']['abstract-events']
        events_data = abstract_events_data.get('events', {})
        spans_sets_data = abstract_events_data.get('span-sets', {})
        for span_set_name, span_set_value in spans_sets_data.items():
            spans = []
            for span_data in span_set_value.get('spans', []):
                spans.append(Span(span_data['string']))
            self.__span_sets[span_set_name] = SpanSet(
                span_set_name=span_set_name,
                spans=spans
            )
        for event_name, event_dict in events_data.items():
            agents = []
            patients = []
            for agent_span_set_id in event_dict['agents']:
                agents.append(self.__span_sets[agent_span_set_id])
            for patient_span_set_id in event_dict['patients']:
                patients.append(self.__span_sets[patient_span_set_id])
            abstract_event = AbstractEvent(
                event_id=event_dict['eventid'],
                helpful_harmful=event_dict['helpful-harmful'],
                material_verbal=event_dict['material-verbal'],
                anchor_span_set=self.__span_sets[event_dict['anchors']],
                agent_span_sets=agents,
                patient_span_sets=patients
            )
            self.add_abstract_event(abstract_event)

    @property
    def abstract_events(self):
        return self.__abstract_events

    @property
    def span_sets(self):
        return self.__span_sets

    @property
    def text(self):
        return self.__text

    @property
    def sent_id(self):
        return self.__sent_id

    @property
    def entry_id(self):
        return self.__entry_id

    # Creates a span set and returns the span set id.  If an identical span set
    # already existed, that span set id is returned instead of creating a new
    # one.
    def add_span_set(self, *, span_strings):
        spans = []
        for span_string in span_strings:
            assert(span_string in self.text)
            spans.append(Span(span_string))
        for ss_id, span_set in self.span_sets.items():
            if spans == span_set.spans:
                return ss_id
        new_ss_id = f'ss-{len(self.span_sets)+1}'
        self.span_sets[new_ss_id] = SpanSet(span_set_name=new_ss_id,
                                            spans=spans)
        return new_ss_id

    # Add a new abstract event that references span sets that already exist on
    # this object
    def add_abstract_event(self, abstract_event):
        # We have to cast to string because MITRE was mixing strings and ints
        key = str(abstract_event.event_id)
        assert(key not in self.abstract_events)
        self.__abstract_events[key] = abstract_event

    def clear_annotation(self):
        self.abstract_events.clear()
        self.span_sets.clear()

    def to_json_dict(self):
        events = {}
        span_sets = {}
        for event_id, event in self.abstract_events.items():
            events[event_id] = event.to_json_dict()
        for ss_id, span_set in self.span_sets.items():
            span_sets[ss_id] = span_set.to_json_dict()
        abstract_events = {
            'events': events,
            'span-sets': span_sets
        }
        annotation_sets = {
            'abstract-events': abstract_events
        }
        data = {
            'annotation-sets': annotation_sets,
            'doc-id': self.__doc_id,
            'entry-id': self.entry_id,
            'segment-text': self.text,
            'segment-type': 'sentence',
            'sent-id': str(self.sent_id)
        }
        return data


class AbstractEvent:
    # Removed SPECIFIED and NOT as they no longer show up as of 8d
    HELPFUL_HARMFUL_TYPES = {'helpful', 'harmful', 'neutral', 'unk'}
    MATERIAL_VERBAL_TYPES = {'material', 'verbal', 'both', 'unk'}

    def __init__(self, *, event_id, helpful_harmful, material_verbal,
                 anchor_span_set, agent_span_sets, patient_span_sets):
        if helpful_harmful not in self.HELPFUL_HARMFUL_TYPES:
            raise RuntimeError(
                f'Unexpected  helpful-harmful value: "{helpful_harmful}"')
        if material_verbal not in self.MATERIAL_VERBAL_TYPES:
            raise RuntimeError(
                f'Unexpected  material-verbal value: "{material_verbal}"')
        self.__event_id = event_id
        self.__helpful_harmful = helpful_harmful
        self.__material_verbal = material_verbal
        self.__anchors = anchor_span_set
        self.__agents = agent_span_sets
        self.__patients = patient_span_sets

    @property
    def agents(self):
        return self.__agents

    @property
    def patients(self):
        return self.__patients

    @property
    def anchors(self):
        return self.__anchors

    @property
    def helpful_harmful(self):
        return self.__helpful_harmful

    @property
    def material_verbal(self):
        return self.__material_verbal

    @property
    def event_id(self):
        return self.__event_id

    def to_json_dict(self):
        data = {
            'agents': sorted([x.ss_id for x in self.agents]),
            'anchors': self.anchors.ss_id,
            'eventid': self.event_id,
            'helpful-harmful': self.helpful_harmful,
            'material-verbal': self.material_verbal,
            'patients': sorted([x.ss_id for x in self.patients])
        }
        return data


class SpanSet:
    def __init__(self, *, span_set_name, spans):
        self.__spans = spans
        self.__ss_id = span_set_name

    @property
    def spans(self):
        return self.__spans

    @property
    def ss_id(self):
        return self.__ss_id

    def to_json_dict(self):
        spans = []
        for span in self.spans:
            spans.append({'string': span.text})
        data = {
            'spans': spans,
            'ssid': self.ss_id
        }
        return data


class Span:
    def __init__(self, text):
        self.__text = text

    def __eq__(self, other):
        if isinstance(other, Span):
            return self.__text == other.__text
        return NotImplemented

    @property
    def text(self):
        return self.__text


def _main(args):
    corpus = Corpus(args.input_file)
    print('Read {} docs'.format(len(corpus.docs)))
    corpus.save(args.output_file)


def _parser_setup():
    parser = argparse.ArgumentParser(
        description="Test ingestion and serialization of MITRE's JSON format")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    return parser


if __name__ == '__main__':
    _main(_parser_setup().parse_args())
