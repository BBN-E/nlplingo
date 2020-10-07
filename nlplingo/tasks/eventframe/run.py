
import sys
import argparse
import os
import json
import logging
import random
import codecs
import copy

from collections import defaultdict

import numpy as np

from nlplingo.embeddings.word_embeddings import load_embeddings
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.eventtrigger.generator import EventTriggerExampleGenerator
from nlplingo.tasks.eventpair.run import evaluate_f1_binary
from nlplingo.tasks.eventpair.run import mAP

from nlplingo.nn.extractor import Extractor
from nlplingo.tasks.eventtrigger.run import generate_trigger_data_feature
from nlplingo.tasks.eventargument.run import generate_argument_data_feature
from nlplingo.tasks.eventtrigger.example import EventTriggerExample
from nlplingo.tasks.eventargument.example import EventArgumentExample
from nlplingo.tasks.eventframe.example import EventFrameExample

logger = logging.getLogger(__name__)


class EventFramePairData(object):
    def __init__(self, pair_examples, data, data_list, label):
        """
        :type pair_examples: list[nlplingo.event.eventframe.example.EventFramePairExample]
        :type data: defaultdict[str, list[numpy.ndarray]]
        :type data_list: list[numpy.ndarray]
        :type label: numpy.ndarray
        """
        self.pair_examples = pair_examples
        self.data = data
        self.data_list = data_list
        self.label = label


def train_eventframepair(params, word_embeddings, trigger_extractor, argument_extractor, eventframepair_extractor):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type eventframepair_extractor: nlplingo.nn.extractor.Extractor
    """

    # prepare dataset for sample generation
    logger.info("Preparing docs")
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
    #test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)
    logger.info("Applying domain")
    #for doc in train_docs + dev_docs + test_docs:
    for doc in train_docs +dev_docs:
        doc.apply_domain(eventframepair_extractor.domain)

    print('#### Generating Training data')
    train_data = generate_pair_data_feature(trigger_extractor, argument_extractor, eventframepair_extractor, train_docs)
    """:type: EventFramePairData"""

    print('#### Generating Dev data')
    dev_data = generate_pair_data_feature(trigger_extractor, argument_extractor, eventframepair_extractor, dev_docs)
    """:type: EventFramePairData"""

    eventframepair_model = eventframepair_extractor.extraction_model
    """:type: nlplingo.nn.eventframepair_model.EventFramePairModel"""
    eventframepair_model.fit_model(train_data.data_list, train_data.label, [], [])     # forgo validation during training epoch, to save compute time

    # Save model data
    if params['save_model']:
        print('==== Saving EventFramePair model ====')
        eventframepair_model.save_keras_model(eventframepair_extractor.model_file)

    # ==== dev data scoring ====
    dev_predictions = eventframepair_model.predict(dev_data.data_list)

    f1_dev = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, 0.5)
    map_dev = mAP(dev_data.label, dev_predictions)

    for f1 in f1_dev:
        print('Dev F1 score: {}\tMAP: {}'.format(f1.to_string(), map_dev))
    with open(params['train.score_file'], 'w') as o:
        for f1 in f1_dev:
            o.write('F1 score: {}\tMAP: {}\n'.format(f1.to_string(), map_dev))


def generate_pair_data_feature(trigger_extractor, argument_extractor, eventframepair_extractor, docs):
    """
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type eventframepair_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """

    trigger_feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator"""
    trigger_example_generator = trigger_extractor.example_generator
    """:type: nlplingo.tasks.eventtrigger.generator.EventTriggerExampleGenerator"""

    argument_feature_generator = argument_extractor.feature_generator
    """:type: nlplingo.tasks.eventargument.feature.EventArgumentFeatureGenerator"""
    argument_example_generator = argument_extractor.example_generator
    """:type: nlplingo.tasks.eventargument.generator.EventArgumentExampleGenerator"""

    eventframepair_feature_generator = eventframepair_extractor.feature_generator
    """:type: nlplingo.tasks.eventframe.feature.EventFramePairFeatureGenerator"""
    eventframepair_example_generator = eventframepair_extractor.example_generator
    """:type: nlplingo.tasks.eventframe.generator.EventFramePairExampleGenerator"""

    eventframepair_examples = eventframepair_example_generator.generate(docs, trigger_feature_generator, argument_feature_generator)
    """:type: list[nlplingo.tasks.eventframe.example.EventFramePairExample]"""


    data = eventframepair_example_generator.examples_to_data_dict(eventframepair_examples, eventframepair_feature_generator.features)

    data_list = [np.asarray(data[k]) for k in eventframepair_feature_generator.features.feature_strings]
    label = np.asarray(data['label'])

    return EventFramePairData(eventframepair_examples, data, data_list, label)


def sample_code_to_generate_events(params, word_embeddings, trigger_extractor, argument_extractor, eventframepair_extractor):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type eventframepair_extractor: nlplingo.nn.extractor.Extractor
    """

    docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)   # this could be a list of SerifXMLs containing <EventMention>, i.e. with triggers and arguments already detected
    """:type: list[nlplingo.text.text_theory.Document]"""

    for doc in docs:
        doc.apply_domain(eventframepair_extractor.domain)

    eventframes_in_docs = defaultdict(list)
    for doc in docs:
        eventframes_in_doc = []
        for sentence in doc.sentences:
            for event in sentence.events:   # transform each event into 1 or more eventframes, depending on the number of agents and patients within that event

                for anchor in event.anchors:
                    # create EventTriggerDatapoint from anchor
                    trigger_example = EventTriggerExample(anchor, sentence, trigger_extractor.domain,
                                                          trigger_extractor.example_generator.extractor_params['embeddings']['vector_size'], anchor.label)
                    """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""

                    # populate 'trigger_example' with features using trigger_extractor.feature_generator
                    trigger_extractor.feature_generator.generate_example(trigger_example, trigger_extractor.hyper_parameters)

                    argument_examples = []
                    """:type: list[nlplingo.tasks.eventargument.example.EventArgumentExample]"""
                    for argument in event.arguments:
                        # create EventArgumentExample from argument
                        argument_example = EventArgumentExample(anchor, argument.entity_mention, sentence, argument_extractor.domain,
                                                       argument_extractor.example_generator.extractor_params, argument_extractor.feature_generator.features, argument_extractor.hyper_parameters,
                                                       argument.label)
                        """:type: nlplingo.tasks.eventargument.example.EventArgumentExample"""

                        # populate 'argument_example' with features using argument_extractor.feature_generator
                        argument_extractor.feature_generator.generate_example(argument_example, sentence.tokens, argument_extractor.hyper_parameters)

                    # and now some sample dummy code to combine trigger and arguments into an EventFrameExample
                    agents = [eg for eg in argument_examples if eg.label.lower() == 'agent']
                    patients = [eg for eg in argument_examples if eg.label.lower() == 'patient']
                    agents_patients = []
                    agents_patients.append(agents[0])
                    agents_patients.append(patients[0])
                    ef_example = EventFrameExample(trigger_example, agents_patients, eventframepair_extractor.example_generator.extractor_params, eventframepair_extractor.hyper_parameters)
                    eventframes_in_doc.append(ef_example)
        eventframes_in_docs[doc.docid].extend(eventframes_in_doc)

    # and then pair up eventframes across different docs to form EventFramePairExample



def sample_code_to_generate_event_embeddings(params, word_embeddings, trigger_extractor, argument_extractor, eventframepair_extractor):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type eventframepair_extractor: nlplingo.nn.extractor.Extractor
    """

    # prepare dataset for sample generation
    logger.info("Preparing docs")
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    logger.info("Applying domain")

    for doc in train_docs:
        doc.apply_domain(eventframepair_extractor.domain)


    # ==== Triggers ====
    (t_examples, t_data, t_data_list, t_label) = generate_trigger_data_feature(
        trigger_extractor.example_generator, train_docs, trigger_extractor.feature_generator)

    t_embeddings = trigger_extractor.get_embeddings(t_examples, t_data_list)
    # t_embeddings is a list of activation layers, where t_embeddings[i] is associated with t_examples[i]
    # once you have t_embeddings, actually using it to train an eventframepair model still take a bit of work.
    # you'll have to introduce it inside eventframe.example.EventFrameExample, and then also use it within
    # nn.eventframepair_model ...

    # we mentioned that the above t_examples generated by NLPLingo (via trigger_extractor.example_generator)
    # might exclude some gold anchors... if those gold anchors are not Verb, Noun, nor Adjective...
    # So we alternatively talked about generating input layer features yourself, given a EventTriggerDatapoint
    # here is an example. I haven't tested the following!
    t_eg = t_examples[0]
    t_eg_data_list = sample_code_to_generate_input_feature_layer(t_eg, trigger_extractor.feature_generator)
    t_eg_embeddings = trigger_extractor.get_embeddings([t_eg], [t_eg_data_list])


    # ==== Arguments ====
    (arg_examples, arg_data, arg_data_list, arg_label) = generate_argument_data_feature(
        argument_extractor.example_generator, train_docs, argument_extractor.feature_generator)

    arg_embeddings = argument_extractor.get_embeddings(arg_examples, arg_data_list)
    # arg_embeddings is a list of activation layers, where arg_embeddings[i] is associated with arg_examples[i]

    # here is an example for event argument. I haven't tested!
    a_eg = arg_examples[0]
    a_eg_data_list = sample_code_to_generate_input_feature_layer(a_eg, argument_extractor.feature_generator)
    a_eg_embeddings = argument_extractor.get_embeddings([a_eg], [a_eg_data_list])


def sample_code_to_generate_input_feature_layer(example, feature_generator):
    """
    :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
    :type feature_generator: nlplingo.tasks.eventtrigger.feature.EventTriggerFeatureGenerator
    """
    data = EventTriggerExampleGenerator.examples_to_data_dict([example], feature_generator.features)
    data_list = [np.asarray(data[k]) for k in feature_generator.features.feature_strings]
    return data_list


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_eventpair, test_eventpair
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)

    load_extractor_models_from_file = False
    if args.mode in {'test_eventframepair'}:
        load_extractor_models_from_file = True

    trigger_extractors = []
    argument_extractors = []
    eventpair_extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractor = Extractor(params, extractor_params, embeddings, load_extractor_models_from_file)
        if extractor.model_type.startswith('event-trigger_'):
            trigger_extractors.append(extractor)
        elif extractor.model_type.startswith('event-argument_'):
            argument_extractors.append(extractor)
        elif extractor.model_type.startswith('event-framepair_'):
            eventpair_extractors.append(extractor)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(extractor.model_type))

    if args.mode == 'train_eventframepair':
        train_eventframepair(params, embeddings, trigger_extractors[0], argument_extractors[0], eventpair_extractors[0])
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))
