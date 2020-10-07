import argparse
import json
import re
from collections import defaultdict, Counter


class HtmlWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.html_header = "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"> <body>"
        self.html_colors = "<b>Colors: </b> <b> <a style=\"color:red\";> anchors </a></b>" + \
                           "<b> <a style=\"color:blue\";> agents </a></b>" + \
                           "<b> <a style=\"color:green\";> patients </a></b></br></br> <hr>"
        self.html_tail = "</body></html>"
        self.html_events = []

    def sanitize_string(self, input):
        return input.replace("&", "&amp;").replace(">", "&gt;").replace("<", "&lt;").replace("\n", "<br />").replace("\t", "&nbsp;&nbsp;&nbsp;")

    def add_html_event(
            self, event_type, segment_text, role_to_span_list, unmatched_spans):
        html_str = "<p> Type: <b>" + event_type + "</b> </p> <p> Sentence: "

        for char_idx in range(0, len(segment_text)):
            char = segment_text[char_idx]
            for role in role_to_span_list.keys():
                for span in role_to_span_list[role]:
                    if span.start == char_idx:
                        html_str = html_str + self.get_html_left(role)
            html_str = html_str + self.sanitize_string(char)
            for role in role_to_span_list.keys():
                for span in role_to_span_list[role]:
                    if span.end == char_idx:
                        html_str = html_str + self.get_html_right()

        # special case for last char in the sentence
        for role in role_to_span_list.keys():
            for span in role_to_span_list[role]:
                if span.end == len(segment_text):
                    html_str = html_str + self.get_html_right()

        # Print any unmatched spans
        if len(unmatched_spans) > 0:
            html_str += '<br><b>{}</b>'.format('<br>'.join(unmatched_spans))

        html_str = html_str + "</p> <hr>"

        self.html_events.append(html_str)

    def write_to_file(self):
        with open(self.filepath, 'w') as file:
            file.write(self.html_header + "\n" +
                       self.html_colors + "\n")
            for html_event in self.html_events:
                file.write(html_event + "\n")
            file.write(self.html_tail + "\n")

    def get_html_left(self, role):
        role_to_color = {
            "patients": "green",
            "agents": "blue",
            "anchors": "red"
        }
        return "<b> <a style=\"color:" + role_to_color[role] + "\";>"

    def get_html_right(self):
        return "</a></b>"


class Span:
    def __init__(self, start, end, string):
        self.start = start
        self.end = end
        self.string = string

    def to_string(self):
        return self.string + "(" + str(self.start) + ", " + str(self.end) + ")"


def main(input_file, output_file):
    annotation = json.loads(open(input_file).read())
    html_writer = HtmlWriter(output_file)
    for entry_id, entry_dict in annotation['entries'].items():
        assert(entry_id == entry_dict['entry-id'])
        segment_text = entry_dict['segment-text']
        abstract_events = entry_dict['annotation-sets']['abstract-events']
        events = abstract_events['events']
        span_sets = abstract_events['span-sets']
        span_set_id_to_span_list = defaultdict(list)
        unmatched_span_set_id_to_strings = defaultdict(list)
        for span_set_id, span_set_dict in span_sets.items():
            for span in span_set_dict['spans']:
                matches_in_segment = re.finditer(re.escape(span['string']),
                                                 segment_text)
                num_matches = 0
                for m in matches_in_segment:
                    num_matches += 1
                    span = Span(m.start(), m.end(), m.group(0))
                    span_set_id_to_span_list[span_set_id].append(span)
                if num_matches == 0:
                    unmatched_span_set_id_to_strings[span_set_id].append(
                        span['string'])

        for event_id, event_dict in events.items():
            role_to_span_list = defaultdict(list)

            # Weirdly, the value of 'eventid' is sometimes an int
            assert(event_id == str(event_dict['eventid']))
            event_type = "{}-{}".format(event_dict['helpful-harmful'],
                                        event_dict['material-verbal'])
            unmatched_spans = []
            for span_set_id in event_dict['agents']:
                assert(span_set_id in span_set_id_to_span_list or
                       span_set_id in unmatched_span_set_id_to_strings)
                # Handle spans that we couldn't match to the sentence
                if span_set_id not in span_set_id_to_span_list:
                    unmatched_spans.append('Unmatched Agent: {}'.format(
                        ', '.join(
                            unmatched_span_set_id_to_strings[span_set_id])))
                for span in span_set_id_to_span_list[span_set_id]:
                    role_to_span_list['agents'].append(span)
            for span_set_id in event_dict['patients']:
                assert (span_set_id in span_set_id_to_span_list or
                        span_set_id in unmatched_span_set_id_to_strings)
                # Handle spans that we couldn't match to the sentence
                if span_set_id not in span_set_id_to_span_list:
                    unmatched_spans.append('Unmatched Patient: {}'.format(
                        ', '.join(
                            unmatched_span_set_id_to_strings[span_set_id])))
                for span in span_set_id_to_span_list[span_set_id]:
                    role_to_span_list['patients'].append(span)

            anchor_span_id = event_dict['anchors']  # Oddly, this is not a list
            assert(isinstance(anchor_span_id, str))
            anchor_span_list = span_set_id_to_span_list[anchor_span_id]
            assert(len(anchor_span_list) > 0)
            for span in anchor_span_list:
                role_to_span_list['anchors'].append(span)

            # Generate HTML
            html_writer.add_html_event(event_type, segment_text,
                                       role_to_span_list, unmatched_spans)

    html_writer.write_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Summarize event annotation in MITRE's JSON format")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
