import os
import json
import logging
import numpy as np

from nlplingo.nn.extractor import Extractor
from nlplingo.tasks.eventargument.run import generate_argument_data_feature, get_predicted_positive_triggers
from nlplingo.embeddings.word_embeddings import load_embeddings
from nlplingo.tasks.eventtrigger.run import apply_positive_training_trigger_filter_to_predictions
from nlplingo.tasks.eventtrigger.run import generate_trigger_data_feature
from nlplingo.tasks.eventrelation.run import generate_eer_data_feature, generate_opennre_triplets_from_candidates
from nlplingo.tasks.eventrelation.postprocess import prefilter, construct_rev_key, print_relation_from_eer_key
from nlplingo.tasks.eventrelation.example import EventEventRelationExample
from nlplingo.text.text_theory import Sentence
from nlplingo.text.text_span import Anchor
logger = logging.getLogger(__name__)
import time

from nlplingo.decoding.prediction_theory import DocumentPrediction, SentencePrediction, EventPrediction, \
    TriggerPrediction, ArgumentPrediction, EventEventRelationPrediction

class TriggerExtractorResultCollection(object):
    def __init__(self, predicted_positive_triggers, trigger_extractor, arg_examples_pt, predicted_role):
        self.predicted_positive_triggers = predicted_positive_triggers
        self.arg_examples_pt = arg_examples_pt
        self.predicted_role = predicted_role
        self.trigger_extractor = trigger_extractor
        self.document_predictions = dict()
        """:type: dict[str, DocumentPrediction]"""

    def prediction_objects_to_json(self):
        d = dict()
        for doc in self.document_predictions.values():
            d[doc.docid] = doc.to_json()
        return d

    def organize_into_prediction_objects(self):
        self.organize_triggers_into_prediction_objects()
        self.organize_arguments_into_prediction_objects()

    def organize_triggers_into_prediction_objects(self):
        for docid in self.predicted_positive_triggers:
            triggers = self.predicted_positive_triggers[docid]
            """:type: list[nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint]"""

            doc_p = DocumentPrediction(docid)
            for trigger in triggers:
                if trigger.sentence.int_pair.to_string() in doc_p.sentences:
                    sentence_p = doc_p.sentences[trigger.sentence.int_pair.to_string()]
                else:
                    sentence_p = SentencePrediction(trigger.sentence.start_char_offset(), trigger.sentence.end_char_offset())
                    doc_p.sentences[trigger.sentence.int_pair.to_string()] = sentence_p

                if trigger.anchor.int_pair.to_string() in sentence_p.events:
                    event_p = sentence_p.events[trigger.anchor.int_pair.to_string()]
                    trigger_p = event_p.trigger
                else:
                    trigger_p = TriggerPrediction(trigger.anchor.start_char_offset(), trigger.anchor.end_char_offset())
                    sentence_p.events[trigger.anchor.int_pair.to_string()] = EventPrediction(trigger_p)

                if trigger.event_type not in trigger_p.labels:
                    trigger_p.labels[trigger.event_type] = trigger.score
                elif trigger_p.labels[trigger.event_type] < trigger.score:
                    trigger_p.labels[trigger.event_type] = trigger.score

            self.document_predictions[docid] = doc_p


    def organize_arguments_into_prediction_objects(self):
        for i, arg in enumerate(self.arg_examples_pt):
            """:type arg: nlplingo.tasks.eventargument.example.EventArgumentExample"""
            role = self.predicted_role[i]

            doc = self.document_predictions[arg.arg0.sentence.docid]
            sentence = doc.sentences[arg.arg0.sentence.int_pair.to_string()]
            event = sentence.events[arg.anchor.int_pair.to_string()]

            if arg.argument.int_pair.to_string() in event.arguments:
                arg_p = event.arguments[arg.argument.int_pair.to_string()]
            else:
                arg_p = ArgumentPrediction(arg.argument.start_char_offset(), arg.argument.end_char_offset())
                arg_p.em_id = arg.argument.id
                event.arguments[arg.argument.int_pair.to_string()] = arg_p

            if role not in arg_p.labels:
                arg_p.labels[role] = arg.score
            elif arg_p.labels[role] < arg.score:
                arg_p.labels[role] = arg.score

class EventEventRelationResultCollection(object):
    def __init__(self,eer_examples_pt,predicted_role, confidences):
        self.eer_examples_pt = eer_examples_pt
        self.predicted_role = predicted_role
        self.document_predictions = dict()
        self.confidences = confidences


    def prediction_objects_to_json(self):
        d = dict()
        for doc in self.document_predictions.values():
            d[doc.docid] = doc.to_json()
        return d

    def organize_into_prediction_objects(self):
        assert len(self.eer_examples_pt) == len(self.predicted_role)
        self.document_predictions = dict()
        for idx in range(len(self.eer_examples_pt)):
            eg = self.eer_examples_pt[idx]
            label = self.predicted_role[idx]
            assert isinstance(eg, EventEventRelationExample)
            sentence = eg.sentence
            assert isinstance(sentence,Sentence)
            docid = sentence.docid
            doc_p = self.document_predictions.setdefault(docid,DocumentPrediction(docid))
            assert isinstance(doc_p,DocumentPrediction)
            left_anchor = eg.anchor1
            assert isinstance(left_anchor,Anchor)
            right_anchor = eg.anchor2
            assert isinstance(right_anchor,Anchor)
            sentence_p = doc_p.sentences.setdefault(sentence.int_pair.to_string(), SentencePrediction(sentence.start_char_offset(), sentence.end_char_offset()))
            assert isinstance(sentence_p,SentencePrediction)

            if left_anchor.int_pair.to_string() not in sentence_p.events:
                trigger_p = TriggerPrediction(left_anchor.start_char_offset(),left_anchor.end_char_offset())
                event_p = sentence_p.events.setdefault(left_anchor.int_pair.to_string(),EventPrediction(trigger_p))
            if right_anchor.int_pair.to_string() not in sentence_p.events:
                trigger_p = TriggerPrediction(right_anchor.start_char_offset(),right_anchor.end_char_offset())
                event_p = sentence_p.events.setdefault(right_anchor.int_pair.to_string(),EventPrediction(trigger_p))

            left_event_p = sentence_p.events[left_anchor.int_pair.to_string()]
            right_event_p = sentence_p.events[right_anchor.int_pair.to_string()]

            event_event_relation_p = sentence_p.event_event_relations.setdefault((left_anchor.int_pair,right_anchor.int_pair),EventEventRelationPrediction(left_event_p,right_event_p))
            assert isinstance(event_event_relation_p,EventEventRelationPrediction)
            event_event_relation_p.labels[label] = self.confidences[idx]

class EventEventRelationResultCollectionConsolidated(object):
    def __init__(self, all_eer_predictions, sent_edt_off_to_sent_dict):
        self.all_eer_predictions = all_eer_predictions
        self.sent_edt_off_to_sent_dict = sent_edt_off_to_sent_dict
        self.learnit_ct = 0
        self.nn_ct = 0

    def prediction_objects_to_json(self):
        d = dict()
        for doc in self.document_predictions.values():
            d[doc.docid] = doc.to_json()
        return d

    def process_neural_prediction(self, actual_item):
        (conf, label, eg) = actual_item
        assert isinstance(eg, EventEventRelationExample)
        sentence = eg.sentence
        assert isinstance(sentence, Sentence)
        docid = sentence.docid
        doc_p = self.document_predictions.setdefault(docid, DocumentPrediction(docid))
        assert isinstance(doc_p, DocumentPrediction)
        left_anchor = eg.anchor1
        assert isinstance(left_anchor, Anchor)
        right_anchor = eg.anchor2
        assert isinstance(right_anchor, Anchor)
        sentence_p = doc_p.sentences.setdefault(sentence.int_pair.to_string(), SentencePrediction(sentence.start_char_offset(), sentence.end_char_offset()))
        assert isinstance(sentence_p, SentencePrediction)

        if left_anchor.int_pair.to_string() not in sentence_p.events:
            trigger_p = TriggerPrediction(left_anchor.start_char_offset(), left_anchor.end_char_offset())
            event_p = sentence_p.events.setdefault(left_anchor.int_pair.to_string(), EventPrediction(trigger_p))
        if right_anchor.int_pair.to_string() not in sentence_p.events:
            trigger_p = TriggerPrediction(right_anchor.start_char_offset(), right_anchor.end_char_offset())
            event_p = sentence_p.events.setdefault(right_anchor.int_pair.to_string(),
                                                   EventPrediction(trigger_p))

        left_event_p = sentence_p.events[left_anchor.int_pair.to_string()]
        right_event_p = sentence_p.events[right_anchor.int_pair.to_string()]

        event_event_relation_p = sentence_p.event_event_relations.setdefault(
            (left_anchor.int_pair, right_anchor.int_pair),
            EventEventRelationPrediction(left_event_p, right_event_p))
        assert isinstance(event_event_relation_p, EventEventRelationPrediction)
        event_event_relation_p.labels[label] = conf
        self.nn_ct += 1

    def organize_into_prediction_objects(self):
        """
        This function applies consolidation (disambiguating between 3 models and resolving head-tail order consistency).
        :return:
        """
        # assert len(self.eer_examples_pt) == len(self.predicted_role)
        self.document_predictions = dict()
        self.learnit_relations = dict()
        logging.info('Number of unique relation keys: %s', len(self.all_eer_predictions))
        ct = 0
        eer_key_set = set()
        for eer_key in self.all_eer_predictions:
            docid = eer_key.split('#')[0]
            if eer_key not in eer_key_set:
                rev_key = construct_rev_key(eer_key)
                eer_key_set.add(eer_key)
                eer_key_set.add(rev_key)

                final_item = dict()
                for model in ['LearnIt', 'nn_model1', 'nn_model2']:
                    model_predictions = None
                    rev_start = len(self.all_eer_predictions[eer_key][model])
                    if rev_start == 0:
                        continue

                    if rev_key in self.all_eer_predictions:
                        model_predictions = self.all_eer_predictions[eer_key][model] + self.all_eer_predictions[rev_key][model]
                    else:
                        model_predictions = self.all_eer_predictions[eer_key][model]
                    ct += len(model_predictions)

                    if model == 'LearnIt':
                        # in this case, pick (an arbitrary choice) the first one
                        item = model_predictions[0]
                    else:
                        # in this case, pick (an arbitrary choice) the prediction with highest confidence
                        sort_idx = sorted(range(len(model_predictions)), key=lambda x : model_predictions[x][0], reverse=True)[0]
                        model_predictions.sort(key = lambda x: x[0], reverse=True)
                        item = (model_predictions[0], sort_idx >= rev_start)
                    final_item.update({model : item})

                # unification logic
                # in the case of a single element, add to appropriate category
                # else, if len == 2: prefer 'bert_mention' or neural prediction
                # if len == 3: prfer 'bert_mention'
                actual_model = None
                if len(final_item) == 1:
                    for key in final_item:
                        actual_model = key
                        if key == 'LearnIt':
                            actual_item = final_item[key]
                            is_rev = False
                            if docid not in self.learnit_relations:
                                self.learnit_relations[docid] = set()
                            self.learnit_relations[docid].add(actual_item[1])
                            self.learnit_ct += 1
                        else:
                            actual_item, is_rev = final_item[key]
                            self.process_neural_prediction(actual_item)

                elif len(final_item) == 2:
                    if 'LearnIt' in final_item:
                        for key in final_item:
                            if key != 'LearnIt':
                                actual_model = key
                                actual_item, is_rev = final_item[key]
                                self.process_neural_prediction(actual_item)
                                break
                    else:
                        actual_model = 'nn_model1'
                        actual_item, is_rev = final_item[actual_model]
                        self.process_neural_prediction(actual_item)

                elif len(final_item) == 3:
                    actual_model = 'nn_model1'
                    actual_item, is_rev = final_item[actual_model]
                    self.process_neural_prediction(actual_item)
                else:
                    raise Exception('final_item not formatted properly')


                # useful debugging statements for printing relations
                """
                if is_rev:
                    print_relation_from_eer_key(rev_key, self.sent_edt_off_to_sent_dict, is_rev=True)
                else:
                    print_relation_from_eer_key(eer_key, self.sent_edt_off_to_sent_dict, is_rev=False)

                logging.debug('final_item %s', final_item)

                if actual_model == 'LearnIt':
                    logging.debug('model %s label %s pattern %s', actual_model, actual_item[0], actual_item[2])
                else:
                    logging.debug('model %s label %s conf %s', actual_model, actual_item[1], actual_item[0])
                """

        logging.info('Number of predictions (before deduplication): %s', ct)
        logging.info('Number of LearnIt predictions added: %s', self.learnit_ct)
        logging.info('Number of neural predictions added: %s', self.nn_ct)

class DocEventAndEventArgFeature(object):
    def __init__(self, docid):
        self.docid = docid
        self.bert = dict()
        self.triggers = dict()
        self.arguments = dict()

    def reduce_arguments(self):
        for args in self.arguments.values():
            for arg_key, vectors in args.items():
                mean_vector = np.mean(vectors,axis=0)
                args[arg_key] = arg_key + (mean_vector,)

    def serialize_to_npz(self, predictions_npz_dir):
        np.savez(
            os.path.join(predictions_npz_dir,
                         self.docid + ".npz"),
            bert=self.bert,
            triggers=self.triggers,
            arguments=self.arguments
        )


class Decoder(object):
    def __init__(self, params):
        self.params = params
        self.event_trigger_extractors = list()
        self.event_argument_extractors = list()
        self.event_event_relation_extractors = list()
        self.model_loaded = False

    def load_model(self):
        self.embeddings = load_embeddings(self.params)

        self.event_trigger_extractors = []
        self.event_argument_extractors = []
        self.event_event_relation_extractors = list()
        """:type: list[nlplingo.model.extractor.Extractor]"""
        for extractor_params in self.params['extractors']:
            extractor = Extractor(self.params, extractor_params, self.embeddings, True)
            if extractor.model_type.startswith('event-trigger_'):
                self.event_trigger_extractors.append(extractor)
            elif extractor.model_type.startswith('event-argument_'):
                self.event_argument_extractors.append(extractor)
            elif extractor.model_type.startswith("event-event-relation_"):
                self.event_event_relation_extractors.append(extractor)
            else:
                logger.warning("Dropping unsupported decoding model type {}".format(extractor.model_type))
        self.model_loaded = True

    def use_preloaded_model(self, trigger_extractors, argument_extractors,event_event_relation_extractors):
        self.model_loaded = False
        self.event_trigger_extractors = trigger_extractors
        self.event_argument_extractors = argument_extractors
        self.event_event_relation_extractors = event_event_relation_extractors
        self.model_loaded = True

    def reload_model(self):
        """
        https://stackoverflow.com/questions/45063602/attempting-to-reset-tensorflow-graph-when-using-keras-failing
        :return:
        """
        self.model_loaded = False
        self.event_trigger_extractors = list()
        self.event_argument_extractors = list()
        self.event_event_relation_extractors = list()
        from keras import backend
        backend.clear_session()
        self.load_model()

    def decode_event_event_relation(self,
                                    test_docs,
                                    all_eer_predictions,
                                    sent_edt_off_to_sent_dict
                                    ):

        for eer_extractor in self.event_event_relation_extractors:
            logging.info('Generating EER examples')
            start = time.time()
            eer_examples, _, _, _ = generate_eer_data_feature(eer_extractor.example_generator, test_docs, eer_extractor.feature_generator)
            decode_triplets = generate_opennre_triplets_from_candidates(eer_examples)
            end = time.time()
            logging.info('EER feature generation took %s seconds', end - start)

            # prefilter examples
            logging.info('Starting prefiltering')
            start = time.time()
            original_indices = list(range(len(eer_examples)))
            prefilter(original_indices, eer_examples)
            eer_examples = [eer_examples[i] for i in original_indices]
            decode_triplets = [decode_triplets[i] for i in original_indices]
            end = time.time()
            logging.info('Prefiltering took %s seconds', end - start)

            logging.info('Starting EER predictions')
            start = time.time()
            all_eer_predictions = eer_extractor.extraction_model.predict_triplets(eer_examples, decode_triplets, all_eer_predictions)
            end = time.time()
            logging.info('EER predictions took %s seconds', end - start)

        return EventEventRelationResultCollectionConsolidated(all_eer_predictions, sent_edt_off_to_sent_dict)

    def decode_argument_all_extractor(self, test_docs, trigger_extractor_result_collection,
                                      doc_id_to_event_and_event_arg_feature=None):
        assert self.model_loaded is True
        if len(self.event_argument_extractors) == 0:
            raise RuntimeError('At least one argument extractor must be specified to decode over arguments.')
        predicted_positive_triggers = trigger_extractor_result_collection.predicted_positive_triggers
        actor_ner_types = {'PER', 'ORG', 'GPE'}
        place_ner_types = {'GPE', 'FAC', 'LOC', 'ORG'}
        time_ner_types = {'TIMEX2.TIME'}

        arg_examples_pt_all = list()
        predicted_role_all = list()

        for test_doc in test_docs:
            if len(predicted_positive_triggers.get(test_doc.docid, list())) > 0:
                for argument_extractor in self.event_argument_extractors:
                    logger.info('Generating argument examples based on predicted triggers')
                    (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                        generate_argument_data_feature(
                            argument_extractor.example_generator, [test_doc], argument_extractor.feature_generator,
                            predicted_triggers=predicted_positive_triggers)
                    if len(arg_examples_pt) == 0:
                        continue

                    logger.info('Predicting arguments based on predicted triggers')
                    argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)
                    pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

                    trigger_extractor = trigger_extractor_result_collection.trigger_extractor

                    if (argument_extractor.emit_vectors
                            and trigger_extractor.emit_vectors):

                        # get layer preceding classification
                        argument_extractor.make_last_layer_model()
                        argument_vectors = (
                            argument_extractor.extraction_model_last_layer
                                .predict(arg_data_list_pt))

                        for i, ex in enumerate(arg_examples_pt):
                            output_vector = argument_vectors[i, :]
                            arg_head = ex.argument.head()
                            a_s_id = ex.arg0.sentence.index
                            a_t_id = arg_head.index_in_sentence
                            a_start = arg_head.start_char_offset()
                            a_end = arg_head.end_char_offset()

                            anchor_head = ex.anchor.head()
                            t_start = anchor_head.start_char_offset()
                            t_end = anchor_head.end_char_offset()

                            arg_key = (t_start, t_end, a_start, a_end)
                            ex_name = argument_extractor.extractor_name

                            vector_data = (
                                doc_id_to_event_and_event_arg_feature
                                    .setdefault(test_doc.docid,
                                                DocEventAndEventArgFeature(
                                                    test_doc.docid)))
                            vector_data.bert[arg_key] = (a_s_id, a_t_id)
                            vector_data.arguments.setdefault(ex_name, {})
                            vector_data.arguments[ex_name].setdefault(
                                arg_key, list()).append(output_vector)

                    assert len(pred_arg_max) == len(arg_examples_pt)
                    for i, predicted_label in enumerate(pred_arg_max):
                        if predicted_label != argument_extractor.domain.get_event_role_index('None'):
                            eg = arg_examples_pt[i]
                            """:type: nlplingo.event.argument.example.EventArgumentExample"""
                            eg.score = argument_predictions_pt[i][predicted_label]
                            predicted_role = argument_extractor.domain.get_event_role_from_index(predicted_label)

                            if (
                                    predicted_role == 'Time' or predicted_role == 'has_time') and eg.argument.label not in time_ner_types:
                                continue
                            if (
                                    predicted_role == 'Place' or predicted_role == 'has_location' or predicted_role == 'has_origin_location' or predicted_role == 'has_destination_location') and eg.argument.label not in place_ner_types:
                                continue
                            if (
                                    predicted_role == 'Actor' or predicted_role == 'Active' or predicted_role == 'Affected' or predicted_role == 'has_active_actor' or predicted_role == 'has_affected_actor') and eg.argument.label not in actor_ner_types:
                                continue
                            arg_examples_pt_all.append(eg)
                            predicted_role_all.append(
                                argument_extractor.domain.get_event_role_from_index(predicted_label))
                            logger.info('PREDICTED-ARGUMENT {} {} {} {} {}'.format(eg.arg0.sentence.docid, predicted_role,
                                                                                    '%.4f' % (eg.score),
                                                                                    eg.argument.start_char_offset(),
                                                                                    eg.argument.end_char_offset()))
        return TriggerExtractorResultCollection(trigger_extractor_result_collection.predicted_positive_triggers,
                                                trigger_extractor_result_collection.trigger_extractor,
                                                arg_examples_pt_all, predicted_role_all)

    def decode_trigger_single_extractor(self, test_docs, trigger_extractor, doc_id_to_event_and_event_arg_feature=None):
        assert self.model_loaded is True
        predicted_positive_triggers_all = dict()

        for test_doc in test_docs:
            # In case this wasn't turned off, and you're testing on the same dataset
            trigger_extractor.example_generator.trigger_candidate_span = None
            logger.info('#### Generating trigger examples for {}'.format(trigger_extractor.extractor_name))
            (trigger_examples, trigger_data, trigger_data_list, trigger_label) = (
                generate_trigger_data_feature(trigger_extractor.example_generator, [test_doc],
                                              trigger_extractor.feature_generator))

            if len(trigger_examples) > 0:
                logger.info('#### Predicting triggers')
                trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)

                using_safelist = trigger_extractor.use_trigger_safelist
                if using_safelist:
                    apply_positive_training_trigger_filter_to_predictions(trigger_examples, trigger_predictions,
                                                                          trigger_extractor)
                if trigger_extractor.emit_vectors:

                    # get layer preceding classification
                    trigger_extractor.make_last_layer_model()
                    trigger_vectors = (
                        trigger_extractor.extraction_model_last_layer.predict(
                            trigger_data_list))

                    for i, ex in enumerate(trigger_examples):
                        output_vector = trigger_vectors[i, :]
                        anchor_head = ex.anchor.head()
                        s_id = ex.sentence.index
                        t_id = anchor_head.index_in_sentence
                        start = anchor_head.start_char_offset()
                        end = anchor_head.end_char_offset()
                        ex_name = trigger_extractor.extractor_name

                        vector_data = (
                            doc_id_to_event_and_event_arg_feature
                                .setdefault(test_doc.docid,
                                            DocEventAndEventArgFeature(
                                                test_doc.docid)))
                        vector_data.bert[start, end] = (s_id, t_id)
                        vector_data.triggers.setdefault(ex_name, {})
                        vector_data.triggers[ex_name][(start, end)] = (
                            start, end, output_vector)

                predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples,
                                                                              trigger_extractor)

                for docid in predicted_positive_triggers:
                    predicted_positive_triggers_all.setdefault(docid, list()).extend(predicted_positive_triggers[docid])
                    for t in predicted_positive_triggers[docid]:
                        """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
                        logger.info(
                            'PREDICTED-ANCHOR {} {} {} {} {}'.format(t.sentence.docid, t.event_type, '%.4f' % t.score,
                                                                     t.anchor.start_char_offset(),
                                                                     t.anchor.end_char_offset()))
        return TriggerExtractorResultCollection(predicted_positive_triggers_all, trigger_extractor, list(), list())

    def decode_trigger_and_argument(self, docs):
        """
        :param docs: List[nlplingo.text.text_theory.Document]
        :return:
        """
        assert self.model_loaded is True
        list_trigger_extractor_result_collection = list()
        doc_id_to_event_and_event_arg_feature = dict()
        for trigger_extractor in self.event_trigger_extractors:
            trigger_extractor_result_collection = self.decode_trigger_single_extractor(docs, trigger_extractor,
                                                                                       doc_id_to_event_and_event_arg_feature)
            trigger_extractor_result_collection_with_argument = self.decode_argument_all_extractor(docs,
                                                                                                   trigger_extractor_result_collection,
                                                                                                   doc_id_to_event_and_event_arg_feature)
            list_trigger_extractor_result_collection.append(trigger_extractor_result_collection_with_argument)

        for feature in doc_id_to_event_and_event_arg_feature.values():
            feature.reduce_arguments()  # get mean vector of repeated arguments

        return list_trigger_extractor_result_collection, doc_id_to_event_and_event_arg_feature

    @staticmethod
    def prediction_objects_to_json(list_trigger_extractor_result_collection):
        d = dict()
        for trigger_extractor_result_collection in list_trigger_extractor_result_collection:
            """:type trigger_extractor_result_collection: TriggerExtractorResultCollection"""
            trigger_extractor_result_collection.organize_into_prediction_objects()
            json_data = trigger_extractor_result_collection.prediction_objects_to_json()
            d[trigger_extractor_result_collection.trigger_extractor.extractor_name] = json_data
        return d

    @staticmethod
    def serialize_prediction_json(list_trigger_extractor_result_collection):
        clusters = dict()
        for trigger_extractor_result_collection in list_trigger_extractor_result_collection:
            predicted_positive_triggers = trigger_extractor_result_collection.predicted_positive_triggers
            for docid in predicted_positive_triggers:
                for idx, t in enumerate(predicted_positive_triggers[docid]):
                    """:type: nlplingo.tasks.eventtrigger.example.EventTriggerDatapoint"""
                    cluster = clusters.setdefault(
                        '{}:{}'.format(trigger_extractor_result_collection.trigger_extractor.extractor_name,
                                       t.event_type), dict())
                    sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))),
                                                  dict())
                    sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
                    sentence['eventType'] = t.event_type
                    sentence['docId'] = t.sentence.docid
                    sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
                    trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
                    trigger_array = trigger.setdefault('trigger', list())
                    trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
                    trigger_array = sorted(list(set(trigger_array)))
                    trigger['trigger'] = trigger_array
                    trigger['score'] = '%.4f' % (t.score)
            for idx, eg in enumerate(trigger_extractor_result_collection.arg_examples_pt):
                predicted_role = trigger_extractor_result_collection.predicted_role[idx]
                cluster = clusters.setdefault(
                    '{}:{}'.format(trigger_extractor_result_collection.trigger_extractor.extractor_name,
                                   eg.anchor.label), dict())
                sentence = cluster.setdefault(str((str(eg.arg0.sentence.docid), str(eg.arg0.sentence.int_pair.to_string()))),
                                              dict())
                if sentence.get('token', None) is None:
                    print("Something is wrong")
                    sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(eg.arg0.sentence.tokens)]
                trigger = sentence.setdefault('trigger_{}'.format(eg.anchor.int_pair.to_string()), dict())
                argument = trigger.setdefault(predicted_role, list())
                # argument.extend([tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens])
                argument_array = [tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]
                argument_d = dict()
                argument_d['start_token_index'] = str(min(argument_array))
                argument_d['end_token_index'] = str(max(argument_array))
                argument_d['score'] = '%.4f' % (eg.score)
                # argument.append((min(argument_array), max(argument_array), '%.4f' % (eg.score)))

                # check whether I have seen this argument offsets before
                to_add = True
                for arg_d in argument:
                    if arg_d['start_token_index'] == argument_d['start_token_index'] and arg_d['end_token_index'] == \
                            argument_d['end_token_index']:
                        to_add = False
                        if float(arg_d['score']) < float(eg.score):
                            arg_d['score'] = '%.4f' % (eg.score)
                        break
                if to_add:
                    argument.append(argument_d)
                # argument = sorted(list(set(argument)))
                trigger[predicted_role] = argument
        return clusters

    @staticmethod
    def serialize_doc_event_and_event_arg_feature_npz(doc_id_to_event_and_event_arg_feature, predictions_npz_dir):
        os.makedirs(predictions_npz_dir, exist_ok=True)
        for event_and_event_arg_feature in doc_id_to_event_and_event_arg_feature.values():
            event_and_event_arg_feature.serialize_to_npz(predictions_npz_dir)
