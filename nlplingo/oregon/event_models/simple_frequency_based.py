# This class serves as an example of how to use abstract_events_data.py to
# load a BP JSON corpus, train on it, extract events from it, and serialize the
# output back into BP JSON.
#
# The actual algorithm used is incredibly stupid, and is meant to just be a
# placeholder.

import argparse
import os
import pickle
import sys
from collections import defaultdict, Counter
from nlplingo.oregon.annotation.abstract_events_data import AbstractEvent, Corpus


class FrequencyBasedEventModel:
    def __init__(self, *, text_to_helpful_harmful_counter,
                 text_to_material_verbal_counter, text_to_arg_role_counter):
        # All text spans stored in the counters are assumed to be lowercased
        self.text_to_helpful_harmful_counter = text_to_helpful_harmful_counter
        self.text_to_material_verbal_counter = text_to_material_verbal_counter
        self.text_to_arg_role_counter = text_to_arg_role_counter

    # Returns a dict of events and a dict of span_sets
    def decode_sentence(self, *, sentence):
        # Naively assume all events in the sentence have the same agent/patient
        best_agent = None
        best_patient = None
        best_agent_count = 0
        best_patient_count = 0
        for token in sentence.text.split():
            norm_token = normalize_text(token)
            if norm_token in self.text_to_arg_role_counter:
                counter = self.text_to_arg_role_counter[norm_token]
                role, count = counter.most_common()[0]
                if role == 'agent' and count > best_agent_count:
                    best_agent = token
                    best_agent_count = count
                elif role == 'patient' and count > best_patient_count:
                    best_patient = token
                    best_patient_count = count
        agent_ss_id, patient_ss_id = None, None
        if best_agent is not None:
            agent_ss_id = sentence.add_span_set(
                span_strings=[best_agent])
        if best_patient is not None:
            patient_ss_id = sentence.add_span_set(
                span_strings=[best_patient])
        # If we have no events in this sentence, the above code means we will
        # still have spans for the agent and patient, but for the purposes of
        # this example script we don't care
        for token in sentence.text.split():
            norm_token = normalize_text(token)
            helpful_harmful = self.pick_helpful_harmful_for_text(
                text=norm_token)
            material_verbal = self.pick_material_verbal_for_text(
                text=norm_token)
            if helpful_harmful is not None and material_verbal is not None:
                event_id = f'event{len(sentence.abstract_events)+1}'
                anchor_ss_id = sentence.add_span_set(span_strings=[token])
                anchor_span_set = sentence.span_sets[anchor_ss_id]
                agent_span_sets, patient_span_sets = [], []
                if agent_ss_id is not None:
                    agent_span_sets.append(sentence.span_sets[agent_ss_id])
                if patient_ss_id is not None:
                    patient_span_sets.append(sentence.span_sets[patient_ss_id])
                abstract_event = AbstractEvent(
                    event_id=event_id,
                    helpful_harmful=helpful_harmful,
                    material_verbal=material_verbal,
                    anchor_span_set=anchor_span_set,
                    agent_span_sets=agent_span_sets,
                    patient_span_sets=patient_span_sets
                )
                sentence.add_abstract_event(abstract_event=abstract_event)

    def pick_helpful_harmful_for_text(self, *, text):
        if text in self.text_to_helpful_harmful_counter:
            counter = self.text_to_helpful_harmful_counter[text]
            return counter.most_common()[0][0]
        else:
            return None

    def pick_material_verbal_for_text(self, *, text):
        if text in self.text_to_material_verbal_counter:
            counter = self.text_to_material_verbal_counter[text]
            return counter.most_common()[0][0]
        else:
            return None


def normalize_text(text):
    return text.lower()


def corpus_count(corpus, text_to_helpful_harmful_counter,
                 text_to_material_verbal_counter, text_to_arg_role_counter):
    for doc_id, doc in sorted(corpus.docs.items(), key=lambda x: x[0]):
        for sentence in doc.sentences:
            for abstract_event in sentence.abstract_events.values():
                for anchor_span in abstract_event.anchors.spans:
                    text = normalize_text(anchor_span.text)
                    text_to_helpful_harmful_counter[text][abstract_event.helpful_harmful] += 1
                    text_to_material_verbal_counter[text][abstract_event.material_verbal] += 1
                for agent_span_set in abstract_event.agents:
                    for agent_span in agent_span_set.spans:
                        text = normalize_text(agent_span.text)
                        text_to_arg_role_counter[text]['agent'] += 1
                for patient_span_set in abstract_event.patients:
                    for patient_span in patient_span_set.spans:
                        text = normalize_text(patient_span.text)
                        text_to_arg_role_counter[text]['patient'] += 1


def get_paths_to_file_in_directory(directory):
    paths = []
    for filename in os.listdir(directory):
        if os.path.isdir(filename):
            raise RuntimeError(
                f'Did not expect nested directories in {directory}')
        paths.append(os.path.join(directory, filename))
    return paths


def train(args):
    text_to_helpful_harmful_counter = defaultdict(Counter)
    text_to_material_verbal_counter = defaultdict(Counter)
    text_to_arg_role_counter = defaultdict(Counter)
    corpus = Corpus(args.input_bp_json_file)
    corpus_count(corpus, text_to_helpful_harmful_counter,
                 text_to_material_verbal_counter, text_to_arg_role_counter)
    model = FrequencyBasedEventModel(
        text_to_helpful_harmful_counter=text_to_helpful_harmful_counter,
        text_to_material_verbal_counter=text_to_material_verbal_counter,
        text_to_arg_role_counter=text_to_arg_role_counter)
    pickle.dump(model, open(args.output_model_file, 'wb'))


def retrain(args):
    model = pickle.load(open(args.input_model_file, 'rb'))
    text_to_helpful_harmful_counter = model.text_to_helpful_harmful_counter
    text_to_material_verbal_counter = model.text_to_material_verbal_counter
    text_to_arg_role_counter = model.text_to_arg_role_counter
    for path in get_paths_to_file_in_directory(args.input_bp_json_dir):
        corpus = Corpus(path)
        corpus_count(corpus, text_to_helpful_harmful_counter,
                     text_to_material_verbal_counter, text_to_arg_role_counter)
    new_model = FrequencyBasedEventModel(
        text_to_helpful_harmful_counter=text_to_helpful_harmful_counter,
        text_to_material_verbal_counter=text_to_material_verbal_counter,
        text_to_arg_role_counter=text_to_arg_role_counter)
    pickle.dump(new_model, open(args.output_model_file, 'wb'))


def decode(args):
    model = pickle.load(open(args.input_model_file, 'rb'))
    for path in get_paths_to_file_in_directory(args.input_bp_json_dir):
        corpus = Corpus(path)
        corpus.clear_annotation()  # Remove any existing event annotation
        for doc_id, doc in sorted(corpus.docs.items(), key=lambda x: x[0]):
            for sentence in doc.sentences:
                model.decode_sentence(sentence=sentence)
        filename = os.path.basename(path)
        corpus.save(os.path.join(args.output_bp_json_dir, filename))


def main(args):
    parser = argparse.ArgumentParser(
        description=("Train, retrain, and decode using a simple frequency-based"
                     "model"))
    subparsers = parser.add_subparsers(
        title="Modes",
        help="You must specify one of these modes to use the model")
    train_parser = subparsers.add_parser(
        'train', help="Train the model from some abstract event data")
    train_parser.set_defaults(func=train)
    train_parser.add_argument('input_bp_json_file')
    train_parser.add_argument('output_model_file')
    decode_parser = subparsers.add_parser(
        'decode', help="Decode some abstracts event data with the model")
    decode_parser.set_defaults(func=decode)
    decode_parser.add_argument('input_model_file')
    decode_parser.add_argument('input_bp_json_dir')
    decode_parser.add_argument('output_bp_json_dir')
    retrain_parser = subparsers.add_parser(
        'retrain', help="Retrain the model from some abstract event data")
    retrain_parser.set_defaults(func=retrain)
    retrain_parser.add_argument('input_model_file')
    retrain_parser.add_argument('input_bp_json_dir')
    retrain_parser.add_argument('output_model_file')

    # Print a help message if we have no arguments
    if len(args) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


if __name__ == '__main__':
    main(sys.argv[1:])
