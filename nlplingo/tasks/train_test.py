from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging

import numpy as np

from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.embeddings.word_embeddings import load_embeddings

from nlplingo.tasks.eventargument.run import generate_argument_data_feature, generate_argument_data_feature_from_serialized
from nlplingo.tasks.eventargument.run import test_argument
from nlplingo.tasks.eventargument.run import train_argument
from nlplingo.tasks.event_domain import EventDomain

from nlplingo.tasks.eventtrigger.run import generate_trigger_data_feature
from nlplingo.tasks.eventtrigger.run import get_predicted_positive_triggers
from nlplingo.tasks.eventtrigger.run import get_predicted_positive_triggers_with_active_learning_metric
from nlplingo.tasks.eventtrigger.run import test_trigger
from nlplingo.tasks.eventtrigger.run import train_trigger_from_file
from nlplingo.tasks.eventtrigger.run import apply_positive_training_trigger_filter_to_predictions
from nlplingo.tasks.eventtrigger.run import active_learning_experiment

from nlplingo.tasks.eventrelation.run import train_eer_from_file, test_eer_with_gold_events_opennre, train_eer_from_json, train_eer_from_json_and_test, train_eer_from_file_pytorch
from nlplingo.tasks.entityrelation.run import train_entity_relation_from_file, generate_entity_relation_data_feature_from_serialized, generate_entity_relation_data_feature
from nlplingo.tasks.sequence.run import decode_sequence_trigger_argument, train_ner, decode_ner, test_ner

from nlplingo.nn.extractor import Extractor
from nlplingo.decoding.decoder import Decoder
from nlplingo.tasks.spanpair.run import train_spanpair

from nlplingo.tasks.cross_task_run import train_bert_mention

logger = logging.getLogger(__name__)


def decode_trigger(params, word_embeddings, trigger_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractors: list[nlplingo.nn.extractor.Extractor] # trigger extractors
    """
    if len(trigger_extractors) == 0:
        raise RuntimeError('At least one trigger extractor must be specified to decode over triggers.')

    #actor_ner_types = set(['PER', 'ORG', 'GPE'])
    #place_ner_types = set(['GPE', 'FAC', 'LOC', 'ORG'])
    #time_ner_types = set(['TIMEX2.TIME'])

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    decoder = Decoder(params)
    decoder.use_preloaded_model(trigger_extractors, list(),list())
    doc_id_to_event_and_event_arg_feature = dict()
    list_trigger_extractor_result_collection = list()
    for trigger_extractor in decoder.event_trigger_extractors:
        trigger_extractor_result_collection = decoder.decode_trigger_single_extractor(test_docs, trigger_extractor,
                                                                                   doc_id_to_event_and_event_arg_feature)
        list_trigger_extractor_result_collection.append(trigger_extractor_result_collection)

    with open(params['predictions_file'], 'w') as fp:
        json.dump(Decoder.serialize_prediction_json(list_trigger_extractor_result_collection), fp, indent=4,
                  sort_keys=True, ensure_ascii=False)

    with open(params['predictions_file']+'.as_obj', 'w') as fp:
        json.dump(Decoder.prediction_objects_to_json(list_trigger_extractor_result_collection), fp, indent=4,
                  sort_keys=True, ensure_ascii=False)

    # Should we write event and event arg feature npz as well?
    docid_set = set(i.docid for i in test_docs)
    if len(docid_set.intersection(doc_id_to_event_and_event_arg_feature.keys())) > 0:
        Decoder.serialize_doc_event_and_event_arg_feature_npz(doc_id_to_event_and_event_arg_feature,
                                                              params['predictions_npz_dir'])



def decode_trigger_argument(params, word_embeddings, trigger_extractors, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractors: list[nlplingo.nn.extractor.Extractor] # trigger extractors
    :type argument_extractors: list[nlplingo.nn.extractor.Extractor] # argument extractors
    """
    # TODO hack until YS has bandwidth to properly integrate after BETTER eval
    if trigger_extractors[0].model_type.startswith('oregon') or argument_extractors[0].model_type.startswith('oregon'):
        from nlplingo.oregon.nlplingo.tasks.sequence.run import decode as oregon_decode
        return oregon_decode(params, trigger_extractors[0].extractor_params, argument_extractors[0].extractor_params, trigger_extractors[0].domain, argument_extractors[0].domain, trigger_extractors[0].hyper_parameters, argument_extractors[0].hyper_parameters)

    if len(trigger_extractors) == 0:
        raise RuntimeError('At least one trigger extractor must be specified to decode over triggers.')

    if len(argument_extractors) == 0:
        raise RuntimeError('At least one argument extractor must be specified to decode over arguments.')

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    # TODO CYS: this is current a hack. This needs to be better factorized and integrated more generically with the existing decode code
    #if trigger_extractors[0].engine == 'transformers' and argument_extractors[0].engine == 'transformers':
    if trigger_extractors[0].model_type.startswith('sequence_') and argument_extractors[0].model_type.startswith('sequence_'):
        return decode_sequence_trigger_argument(params, trigger_extractors[0], argument_extractors[0], test_docs)
        #return decode_event_transformer(params, trigger_extractors[0], argument_extractors[0], test_docs)

    decoder = Decoder(params)
    decoder.use_preloaded_model(trigger_extractors,argument_extractors,list())
    list_trigger_extractor_result_collection,doc_id_to_event_and_event_arg_feature = decoder.decode_trigger_and_argument(test_docs)

    with open(params['predictions_file'], 'w') as fp:
        json.dump(Decoder.serialize_prediction_json(list_trigger_extractor_result_collection), fp, indent=4, sort_keys=True,ensure_ascii=False)

    with open(params['predictions_file']+'.as_obj', 'w') as fp:
        json.dump(Decoder.prediction_objects_to_json(list_trigger_extractor_result_collection), fp, indent=4,
                  sort_keys=True, ensure_ascii=False)

    # Should we write event and event arg feature npz as well?
    docid_set = set(i.docid for i in test_docs)
    if len(docid_set.intersection(doc_id_to_event_and_event_arg_feature.keys())) > 0:
        Decoder.serialize_doc_event_and_event_arg_feature_npz(doc_id_to_event_and_event_arg_feature,params['predictions_npz_dir'])


def decode_argument(params, word_embeddings, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type argument_extractors: list[nlplingo.nn.extractor.Extractor] # argument extractors
    """
    if len(argument_extractors) == 0:
        raise RuntimeError('At least one argument extractor must be specified to decode over arguments.')

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    # TODO CYS: this is current a hack. This needs to be better factorized and integrated more generically with the existing decode code
    if argument_extractors[0].engine == 'transformers':
        return decode_event_transformer_using_gold_trigger(params, argument_extractors[0], test_docs)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)
    parser.add_argument('--serialize_list', default=None)
    parser.add_argument('--k_partitions', type=int, default=None)
    parser.add_argument('--partition_id', type=int, default=None)

    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    load_extractor_models_from_file = False
    if args.mode in {'test_trigger', 'test_argument', 'decode_trigger_argument', 'decode_trigger', 'decode_trigger_argument_for_active_learning'}:
        load_extractor_models_from_file = True

    trigger_extractors = []
    argument_extractors = []
    eer_extractors = []
    entity_relation_extractors = []
    ner_extractors = []
    entitycoref_extractors = []
    eventcoref_extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        embeddings = load_embeddings(extractor_params)
        extractor = Extractor(params, extractor_params, embeddings, load_extractor_models_from_file)
        if extractor.model_type.startswith('event-trigger_') or (extractor.task is not None and extractor.task == 'event-trigger'):
            trigger_extractors.append(extractor)
        elif extractor.model_type.startswith('event-argument_') or (extractor.task is not None and extractor.task == 'event-argument'):
            argument_extractors.append(extractor)
        elif extractor.model_type.startswith('event-event-relation_'):
            eer_extractors.append(extractor)
        elif extractor.model_type.startswith('entity-entity-relation_'):
            entity_relation_extractors.append(extractor)
        elif extractor.task == 'ner':
            ner_extractors.append(extractor)
        elif extractor.model_type.startswith('entitycoref_'):
            entitycoref_extractors.append(extractor)
        elif extractor.model_type.startswith('eventcoref_'):
            eventcoref_extractors.append(extractor)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(extractor.model_type))

    if 'domain_ontology.scoring' in params:
        scoring_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology.scoring'), 'scoring')
    else:
        scoring_domain = None

    if args.mode == 'train_trigger_from_file':
        train_trigger_from_file(params, embeddings, trigger_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'train_argument_from_file':
        train_argument(params, embeddings, argument_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'train_argument_bert_mention':
        train_bert_mention(params, embeddings, argument_extractors[0], args.serialize_list, generate_argument_data_feature, generate_argument_data_feature_from_serialized, args.k_partitions, args.partition_id)
    elif args.mode == 'train_entity_bert_mention':
        train_bert_mention(params, embeddings, entity_relation_extractors[0], args.serialize_list, generate_entity_relation_data_feature, generate_entity_relation_data_feature_from_serialized, args.k_partitions, args.partition_id)
    elif args.mode == 'train_evevr_from_file':
        train_eer_from_file(params, embeddings, eer_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'train_eer_from_txt':
        train_eer_from_json(params, eer_extractors[0])
    elif args.mode == 'train_eer_from_txt_and_test':
        train_eer_from_json_and_test(params, eer_extractors[0])
    elif args.mode == 'train_eer_pytorch':
        train_eer_from_file_pytorch(params, embeddings, eer_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'train_entity_relation_from_file':
        train_entity_relation_from_file(params, embeddings, entity_relation_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'test_trigger':
        test_trigger(params, embeddings, trigger_extractors[0])
    elif args.mode == 'test_eer_opennre':
        test_eer_with_gold_events_opennre(params, embeddings, eer_extractors)
    elif args.mode == 'train_argument':
        train_argument(params, embeddings, argument_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'test_argument':
        test_argument(params, embeddings, trigger_extractors[0], argument_extractors[0], scoring_domain)
    elif args.mode == 'decode_trigger_argument':
        decode_trigger_argument(params, embeddings, trigger_extractors, argument_extractors)
    elif args.mode == 'decode_trigger':
        decode_trigger(params, embeddings, trigger_extractors)
    elif args.mode == 'decode_argument':
        decode_argument(params, embeddings, argument_extractors)
    # elif args.mode == 'decode_trigger_argument_for_active_learning':
    #     decode_trigger_argument_for_active_learning(params, embeddings, trigger_extractors, argument_extractors)
    elif args.mode == 'active_learning_experiment':
        active_learning_experiment(params, embeddings, trigger_extractors[0])
    elif args.mode == 'train_ner':
        train_ner(params, ner_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'decode_ner':
        decode_ner(params, ner_extractors[0])
    elif args.mode == 'test_ner':
        test_ner(params, ner_extractors[0])
    elif args.mode == 'train_entitycoref':
        train_spanpair(params, embeddings, entitycoref_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    elif args.mode == 'train_eventcoref':
        train_spanpair(params, embeddings, eventcoref_extractors[0], args.serialize_list, args.k_partitions, args.partition_id)
    # elif args.mode == 'decode_event_transformer':
    #     decode_event_transformer(params, trigger_extractors[0], argument_extractors[0])
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))
