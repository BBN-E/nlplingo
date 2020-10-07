from nlplingo.common.utils import IntPair
#from nlplingo.tasks.sequence.run import convert_frames_to_prediction_theories, get_matches_from_labeled_sequence
from nlplingo.tasks.sequence.utils import convert_frames_to_prediction_theories, get_matches_from_labeled_sequence
from nlplingo.text.text_span import LabeledTextSpan, LabeledTextFrame

from nlplingo.oregon.nlplingo.tasks.sequence.generator import PipelineGenerator, get_arguments
from nlplingo.oregon.nlplingo.tasks.sequence.ED_trainer import EDTrainer
from nlplingo.oregon.nlplingo.tasks.sequence.argument_trainer import ArgumentTrainer

#from python.clever.event_models.uoregon.models.pipeline._01.iterators import PipelineIterator, get_arguments
#from python.clever.event_models.uoregon.models.pipeline._01.local_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *
import torch
#from nlplingo.oregon.event_models.uoregon.define_opt import logger
#from datetime import datetime
from nlplingo.oregon.event_models.uoregon.tools.corpus_utils import Corpus, LANG_SET, AbstractEvent, get_ori_string, \
    strip_punctuations
from nlplingo.oregon.event_models.uoregon.tools.xlmr import xlmr_tokenizer


class PipelineTrainer(object):
    def __init__(self, opt, logger, biw2v_map, examples, docs, trigger_label_map, argument_label_map):
        #assert lang in LANG_SET, 'Unsupported language! Select from: {}'.format(LANG_SET)
        self.opt = opt
        self.logger = logger

        #self.lang = lang

        print('trigger_label_map=', trigger_label_map)
        print('argument_label_map=', argument_label_map)

        self.trigger_label_map = trigger_label_map
        self.argument_label_map = argument_label_map

        # load config files for trigger and argument, as dict 'ED' and 'argument'. These are printed below
        config = self.load_ckpt_configs()

        """
        print(config.get('ED'))             # 'ED' is just a dict
        {'biw2v_map_dir': 'resources/biw2v_map', 'datapoint_dir': 'datapoints',
         'stanford_resource_dir': 'resources/stanford', 'log_dir': 'logs', 'log_name': 'train.log.ed', 'batch_size': 16,
         'upos_dim': 30, 'xpos_dim': 30, 'deprel_dim': 30, 'dist_dim': 30, 'ner_dim': 30, 'num_last_layer_xlmr': 1,
         'hidden_dim': 200, 'cross_valid': '', 'data': 'abstract', 'max_grad_norm': 5.0, 'optim': 'adam', 'lr': 5e-06,
         'num_epoch': 5, 'trainer': 'trigger', 'seed': 2020, 'context_layer': 'lstm', 'gcn_dropout': 0.5,
         'prune_tree': 0, 'model': 'pipeline-01', 'use_biw2v': 0, 'finetune_biw2v': 0, 'self_att_layers': 6,
         'self_att_heads': 1, 'self_att_d_qkv': 200, 'self_att_dropout': 0.1, 'lstm_by_satt': 0, 'lstm_add_satt': 0,
         'lambda_mix': 0.8, 'lstm_layers_trigger': 4, 'lstm_layers_entity': 1, 'lstm_layers_event': 1,
         'use_dep_edge': 0, 'use_cased_entity': 1, 'use_elmo': 0, 'do_exp': 'default',
         'train_file': 'app/train_data_dir',
         'dev_file': 'datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json', 'test_file': None,
         'output_format': 'json', 'output_file': None, 'input_lang': 'english', 'data_map': None, 'inhouse_eval': 0,
         'ensemble_mode': 0, 'ensemble_seeds': ['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025'],
         'finetune_xlmr': 1, 'dropout_xlmr': 0.5, 'train_ED': 1, 'train_argument': 0, 'get_perf_of_separate_models': 0,
         'edge_lambda': 0.1, 'use_dep2sent': 0, 'xlmr_version': 'xlmr.base', 'xlmr_model_dir': 'models/xlmr.base',
         'position_embed_for_satt': 1, 'ED_eval_epoch': 4, 'argument_eval_epoch': 4, 'output_offsets': 1,
         'train_strategy': 'retrain.add-all', 'delete_nonbest_ckpts': 1, 'docker_run': 0, 'hidden_eval': 0,
         'bad_threshold': 0.4, 'readers_mode': 1, 'use_ner': 0, 'initialize_with_pretrained': 0, 'train_on_arb': 0,
         'remove_incomplete': 0, 'co_train_lambda': 0, 'finetune_on_arb': 0, 'num_first_xlmr_layers': 7,
         'save_last_epoch': 1, 'mode': 'train_trigger_from_docs', 'params': 'nlplingo/trigger/params/train.params',
         'train_is_dir': True, 'test_is_dir': False,
         'finetuned_xlmr_layers': ['xlmr_embedding.model.decoder.sentence_encoder.embed_tokens',
                                   'xlmr_embedding.model.decoder.sentence_encoder.embed_positions',
                                   'self_att.attention_layers', 'gcn_layer', 'biw2v_embedding',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.0.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.1.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.2.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.3.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.4.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.5.'], 
         'biw2v_size': 354186,
         'biw2v_vecs': array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,        
                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                               [ 8.26119033e-01,  3.68800311e-01,  8.69561242e-01, ...,
                                 2.70505650e-01,  2.05427664e-01,  2.01526267e-01],
                               [ 1.33400000e-03,  1.47300000e-03, -1.27700000e-03, ...,
                                -4.37000000e-04, -5.52000000e-04,  1.02400000e-03],
                               ...,
                               [-1.15833000e-01, -8.17270000e-02, -5.58370000e-02, ...,
                                -1.59482000e-01, -3.43660000e-02,  6.65400000e-03],
                               [-3.82970000e-02, -5.19210000e-02, -7.23600000e-02, ...,
                                -1.40313000e-01,  1.73640000e-02,  1.28790000e-02],
                               [-1.11085000e-01, -4.86380000e-02, -8.37620000e-02, ...,
                                -1.55592000e-01,  6.28500000e-03,  3.66210000e-02]]), 
        'device': device(type='cuda')}

        print(config.get('argument'))       # 'argument' is just a dict
        {'biw2v_map_dir': 'resources/biw2v_map', 'datapoint_dir': 'datapoints',
         'stanford_resource_dir': 'resources/stanford', 'log_dir': 'logs', 'log_name': 'train.log.arg',
         'batch_size': 16, 'upos_dim': 30, 'xpos_dim': 30, 'deprel_dim': 30, 'dist_dim': 30, 'ner_dim': 30,
         'num_last_layer_xlmr': 1, 'hidden_dim': 200, 'cross_valid': '', 'data': 'abstract', 'max_grad_norm': 5.0,
         'optim': 'adam', 'lr': 5e-06, 'num_epoch': 5, 'trainer': 'trigger', 'seed': 2020, 'context_layer': 'lstm',
         'gcn_dropout': 0.5, 'prune_tree': 0, 'model': 'pipeline-01', 'use_biw2v': 0, 'finetune_biw2v': 0,
         'self_att_layers': 6, 'self_att_heads': 1, 'self_att_d_qkv': 200, 'self_att_dropout': 0.1, 'lstm_by_satt': 0,
         'lstm_add_satt': 0, 'lambda_mix': 0.8, 'lstm_layers_trigger': 4, 'lstm_layers_entity': 1,
         'lstm_layers_event': 1, 'use_dep_edge': 0, 'use_cased_entity': 1, 'use_elmo': 0, 'do_exp': 'default',
         'train_file': 'app/train_data_dir',
         'dev_file': 'datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json', 'test_file': None,
         'output_format': 'json', 'output_file': None, 'input_lang': 'english', 'data_map': None, 'inhouse_eval': 0,
         'ensemble_mode': 0, 'ensemble_seeds': ['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025'],
         'finetune_xlmr': 1, 'dropout_xlmr': 0.5, 'train_ED': 0, 'train_argument': 1, 'get_perf_of_separate_models': 0,
         'edge_lambda': 0.1, 'use_dep2sent': 0, 'xlmr_version': 'xlmr.base', 'xlmr_model_dir': 'models/xlmr.base',
         'position_embed_for_satt': 1, 'ED_eval_epoch': 4, 'argument_eval_epoch': 4, 'output_offsets': 1,
         'train_strategy': 'retrain.add-all', 'delete_nonbest_ckpts': 1, 'docker_run': 0, 'hidden_eval': 0,
         'bad_threshold': 0.4, 'readers_mode': 1, 'use_ner': 0, 'initialize_with_pretrained': 0, 'train_on_arb': 0,
         'remove_incomplete': 0, 'co_train_lambda': 0, 'finetune_on_arb': 0, 'num_first_xlmr_layers': 7,
         'save_last_epoch': 1, 'mode': 'train_argument_from_docs', 'params': 'nlplingo/argument/params/train.params',
         'train_is_dir': True, 'test_is_dir': False,
         'finetuned_xlmr_layers': ['xlmr_embedding.model.decoder.sentence_encoder.embed_tokens',
                                   'xlmr_embedding.model.decoder.sentence_encoder.embed_positions',
                                   'self_att.attention_layers', 'gcn_layer', 'biw2v_embedding',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.0.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.1.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.2.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.3.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.4.',
                                   'xlmr_embedding.model.decoder.sentence_encoder.layers.5.'], 
         'biw2v_size': 354186,
         'biw2v_vecs': array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,        
                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                               [ 8.26119033e-01,  3.68800311e-01,  8.69561242e-01, ...,
                                 2.70505650e-01,  2.05427664e-01,  2.01526267e-01],
                               [ 1.33400000e-03,  1.47300000e-03, -1.27700000e-03, ...,
                                -4.37000000e-04, -5.52000000e-04,  1.02400000e-03],
                               ...,
                               [-1.15833000e-01, -8.17270000e-02, -5.58370000e-02, ...,
                                -1.59482000e-01, -3.43660000e-02,  6.65400000e-03],
                               [-3.82970000e-02, -5.19210000e-02, -7.23600000e-02, ...,
                                -1.40313000e-01,  1.73640000e-02,  1.28790000e-02],
                               [-1.11085000e-01, -4.86380000e-02, -8.37620000e-02, ...,
                                -1.55592000e-01,  6.28500000e-03,  3.66210000e-02]]), 
        'device': device(type='cuda')}
        """

        # trigger: create trigger model and optimizer
        # argument: create argument model and optimizer
        self.trainer = {
            'ED': EDTrainer(config['ED'], self.logger, biw2v_map, eval_mode=True, label_map=trigger_label_map),  # TODO call the one in tasks.sequence
            'argument': ArgumentTrainer(config['argument'], self.logger, biw2v_map, eval_mode=True, label_map=argument_label_map)  # TODO call the one in tasks.sequence
        }

        self.trainer['ED'].model.eval()
        self.trainer['argument'].model.eval()

        self.test_iterator = PipelineGenerator(
            self.trainer['ED'].model.xlmr_embedding, examples, docs, biw2v_map
            #data_path=os.path.join(opt['datapoint_dir'],
            #                       opt['data_map']['pipeline']['test'].format(get_data_dir(self.opt['data'])))
        )
        print('Pipeline : Test data: {}'.format(self.test_iterator.num_examples))

        #print('Ingesting test_file=', test_file)
        #self.output_corpus = Corpus(filepath=test_file, lang='arabic', parsing=False)
        #self.output_corpus.clear_annotation()

        #self.impact_inverse_map = dict([(v, k) for k, v in IMPACT_MAP.items()])
        #self.effect_inverse_map = dict([(v, k) for k, v in EFFECT_MAP.items()])
        #self.event_inverse_map = dict([(v, k) for k, v in EVENT_MAP.items()])
        self.event_inverse_map = dict([(v, k) for k, v in trigger_label_map.items()])

        # print opt to screen
        cout = '*' * 100 + '\n'
        cout += 'Opt:\n'
        for arg in sorted(opt):
            if arg not in ['biw2v_vecs']:
                cout += '{}: {}\n'.format(arg, opt[arg])
        cout += '*' * 100 + '\n'
        print(cout)
        logger.info(cout)

        cout = '*' * 100 + '\n'
        cout += 'Opt:\n'
        for arg in sorted(opt):
            if arg not in ['biw2v_vecs']:
                cout += '{}: {}\n'.format(arg, opt[arg])
        cout += '*' * 100 + '\n'
        print(cout)
        logger.info(cout)

        #######
        self.docs = docs
        self.frame_annotations = dict()
        for doc in docs:
            l = [[] for _ in range(len(doc.sentences))]
            self.frame_annotations[doc.docid] = l
            assert len(l) == len(doc.sentences)

    def load_ckpt_configs(self):
        # When you specify a particular ED_eval_epoch and a particular argument_eval_epoch, this code will try to find
        # that for you. If cannot find, will regress to the first checkpoint it can find.
        ED_config_path = os.path.join(self.opt['trigger_model_dir'], 'ED.config')
        ED_config = read_json(ED_config_path)
        #ED_config['trigger_model_dir'] = self.opt['trigger_model_dir']

        argument_config_path = os.path.join(self.opt['argument_model_dir'], 'argument.config')
        argument_config = read_json(argument_config_path)
        #argument_config['argument_model_dir'] = self.opt['argument_model_dir']

        for param in ['biw2v_vecs', 'device', 'ED_eval_epoch', 'argument_eval_epoch', 'train_on_arb']:
            ED_config[param] = self.opt[param]
            argument_config[param] = self.opt[param]
        for param in self.opt:
            if param not in ED_config:
                ED_config[param] = self.opt[param]
            if param not in argument_config:
                argument_config[param] = self.opt[param]
        return {'ED': ED_config, 'argument': argument_config}

    # def load_ckpt_configs(self):
    #     # When you specify a particular ED_eval_epoch and a particular argument_eval_epoch, this code will try to find
    #     # that for you. If cannot find, will regress to the first checkpoint it can find.
    #     ED_config_path = os.path.join('checkpoints',
    #                                   'ED.epoch-{}.config'.format(self.opt['ED_eval_epoch']))
    #     argument_config_path = os.path.join('checkpoints',
    #                                         'argument.epoch-{}.config'.format(self.opt['argument_eval_epoch']))
    #
    #     if not os.path.exists(ED_config_path) or not os.path.join(argument_config_path):
    #         ED_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
    #                            len(re.findall(
    #                                r'ED[.]epoch[-]\d+[.]config',
    #                                fname)) > 0]
    #         argument_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
    #                                  len(re.findall(
    #                                      r'argument[.]epoch[-]\d+[.]config',
    #                                      fname)) > 0]
    #         if len(ED_config_paths) * len(argument_config_paths) == 0:
    #             print('Not enough checkpoints!')
    #             if self.opt['docker_run'] == 0:
    #                 print('You might have not renamed checkpoint files from * -> *.hideval.simulated')
    #             exit(1)
    #         else:
    #             print('{} and {} are not found!\nUsing {} and {} instead!'.format(
    #                 ED_config_path,
    #                 argument_config_path,
    #                 ED_config_paths[0],
    #                 argument_config_paths[0]
    #             ))
    #             ED_config_path = ED_config_paths[0]
    #             argument_config_path = argument_config_paths[0]
    #             self.opt['ED_eval_epoch'] = int(
    #                 re.findall(r'epoch[-](\d+)[.]config',
    #                            ED_config_path)[0])
    #             self.opt['argument_eval_epoch'] = int(
    #                 re.findall(r'epoch[-](\d+)[.]config',
    #                            argument_config_path)[0])
    #     # Just ensure that by now, ED_config_path points to something like: ED.epoch-4.config
    #     # Just ensure that by now, argument_config_path points to something like: argument.epoch-4.config
    #
    #     ED_config = read_json(ED_config_path)
    #     argument_config = read_json(argument_config_path)
    #     for param in ['biw2v_vecs', 'device', 'ED_eval_epoch', 'argument_eval_epoch', 'train_on_arb']:
    #         ED_config[param] = self.opt[param]
    #         argument_config[param] = self.opt[param]
    #     for param in self.opt:
    #         if param not in ED_config:
    #             ED_config[param] = self.opt[param]
    #         if param not in argument_config:
    #             argument_config[param] = self.opt[param]
    #     return {'ED': ED_config, 'argument': argument_config}

    def eval(self):
        self.trainer['ED'].load_saved_model()
        self.trainer['argument'].load_saved_model()

        for b_id, inputs in enumerate(self.test_iterator):
            inputs = move_to_device(inputs, self.opt['device'])

            ED_preds, ED_probs = self.trainer['ED'].predict(inputs)

            argument_inputs, eids, trigger_words, event_types, event_probs = self.get_inputs_for_argument_model(inputs,
                                                                                                                ED_preds,
                                                                                                                ED_probs)
            if argument_inputs is not None:
                argument_preds, argument_probs = self.trainer['argument'].predict(argument_inputs)
                self.record_predictions_nlplingo(eids, trigger_words, event_types, event_probs, argument_preds, argument_probs)
                #self.record_predictions(eids, trigger_words, event_types, event_probs, argument_preds, argument_probs)

        #print('Writing predictions to file: {}'.format(self.opt['output_file']))    # directly producing a BP file
        #self.output_corpus.save(output_file=self.opt['output_file'])
        #self.logger.info('Writing predictions to file: {}'.format(self.opt['output_file']))

        print('Converting frames to prediction theories')
        document_predictions = convert_frames_to_prediction_theories(self.frame_annotations, self.docs)
        # serialize out document_predictions
        d = dict()
        d['trigger'] = dict()
        for doc in document_predictions:
            d['trigger'][doc.docid] = doc.to_json()
        print('Writing out to predictions_file')
        with open(self.opt['predictions_file'], 'w', encoding='utf-8') as fp:
            json.dump(d, fp, indent=4, sort_keys=True, ensure_ascii=False)

    # this very much mirrors nlplingo.tasks.sequence.run.decode_trigger_argument()
    def record_prediction_on_example(self, example, argument_bio, anchor_label, anchor_start_token_index, anchor_end_token_index):
        anchor_text = ' '.join(example.words[anchor_start_token_index: anchor_end_token_index + 1])
        anchor_span = LabeledTextSpan(IntPair(None, None), anchor_text, anchor_label)
        anchor_span.start_token_index = anchor_start_token_index
        anchor_span.end_token_index = anchor_end_token_index

        argument_spans = []
        for (start_token_index, end_token_index, argument_label) in get_matches_from_labeled_sequence(argument_bio):
            # TODO we want the argument_text to be raw text, not concatenate of tokens
            argument_text = ' '.join(example.words[start_token_index: end_token_index + 1])
            argument_span = LabeledTextSpan(IntPair(None, None), argument_text, argument_label)
            argument_span.start_token_index = start_token_index
            argument_span.end_token_index = end_token_index
            argument_spans.append(argument_span)

        labeled_frame = LabeledTextFrame([anchor_span], argument_spans)
        self.frame_annotations[example.docid][example.sentence_index].append(labeled_frame)


    def record_predictions_nlplingo(self, eids, trigger_toks_list, event_types, event_probs, argument_preds, argument_probs):
        eids = eids.data.cpu().numpy()
        for k, eid in enumerate(eids):
            ori_example = self.test_iterator.id2ori_example[eid]
            """:type: nlplingo.tasks.sequence.example.SequenceExample"""

            trigger_toks = trigger_toks_list[k]
            event_type = event_types[k]

            tag_ids = argument_preds[k]
            actual_length = len(ori_example.words) - 2
            tag_ids = tag_ids.long().data.cpu().numpy()
            tag_ids = tag_ids[: actual_length]
            tags = [self.trainer['argument'].id2tag[tag_id] for tag_id in tag_ids]

            self.record_prediction_on_example(ori_example, tags, event_type, trigger_toks[0], trigger_toks[-1])


    def record_predictions(self, eids, trigger_toks_list, event_types, event_probs, argument_preds, argument_probs):
        eids = eids.data.cpu().numpy()
        for k, eid in enumerate(eids):
            ori_example = self.test_iterator.id2ori_example[eid]
            """:type: nlplingo.tasks.sequence.example.SequenceExample"""

            trigger_toks = trigger_toks_list[k]
            arg_preds = argument_preds[k]

            """
            print('ori_example.sentence.tokens={}'.format(' '.join(token.text for token in ori_example.sentence.tokens)))
            print('trigger_toks=', trigger_toks)
            print('arg_preds=', arg_preds)
            print('event_types[k]=', event_types[k])
            
            ori_example.sentence.tokens=The Brotherhood was officially dissolved by Egypt 's military rulers in 1954 , but registered itself as a non-governmental organization in March in a response to a court case brought by opponents of the group who were contesting its legality .
            trigger_toks= [4]
            arg_preds= tensor([1, 3, 4, 4, 4, 4, 0, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], device='cuda:0')
            event_types[k]= harmful.material
            """

            ###################### extract texts ########################
            _, _, agent_norm_offsets, patient_norm_offsets = get_arguments(self.trainer['argument'].id2tag,
                                                                           arg_preds, ori_example)
            ############## recover original strings and offsets ##########
            ############## OFFSETS ARE OPEN INTERVALS ####################
            # TODO explore the following in more details
            #anchor_norm_offset = [
            #    ori_example['span'][int(trigger_toks[0])][0], ori_example['span'][int(trigger_toks[-1])][
            #                                                      1] + 1]  # (x, y + 1)
            #anchor_offset, anchor_word = get_ori_string(ori_example, norm_offset=anchor_norm_offset)

            anchor_start_offset = ori_example.sentence.tokens[trigger_toks[0]].start_char_offset()
            anchor_end_offset = ori_example.sentence.tokens[trigger_toks[-1]].end_char_offset()
            anchor_offset = [anchor_start_offset, anchor_end_offset]
            anchor_word = ori_example.sentence.get_text(anchor_start_offset, anchor_end_offset)

            # -------------------------------
            agents, patients = [], []
            agent_offsets, patient_offsets = {}, {}
            for agent in agent_norm_offsets:
                # agent_norm_offset = agent_norm_offsets[agent]
                # agent_ori_offset, agent_ori_string = get_ori_string(ori_example, norm_offset=agent_norm_offset)

                agent_ori_offset = agent_norm_offsets[agent]
                agent_ori_string = ori_example.sentence.get_text(agent_ori_offset[0], agent_ori_offset[-1])

                if len(agent_ori_string) > 0:
                    agents.append(agent_ori_string)
                    agent_offsets[agent_ori_string] = agent_ori_offset

            for patient in patient_norm_offsets:
                # patient_norm_offset = patient_norm_offsets[patient]
                # patient_ori_offset, patient_ori_string = get_ori_string(ori_example, norm_offset=patient_norm_offset)

                patient_ori_offset = patient_norm_offsets[patient]
                patient_ori_string = ori_example.sentence.get_text(patient_ori_offset[0], patient_ori_offset[-1])

                if len(patient_ori_string) > 0:
                    patients.append(patient_ori_string)
                    patient_offsets[patient_ori_string] = patient_ori_offset
            ###############################################################
            agents = list(set(agents))
            patients = list(set(patients))

            agents = [agent for agent in agents if len(agent.strip()) > 0]
            patients = [patient for patient in patients if len(patient.strip()) > 0]

            # event_type = event_types[k]
            # #event_type = event_type.split('-')[1]	# because my event type looks like: B-x.y
            # event_prob = event_probs[k]
            # argument_prob = argument_probs[k]
            # if event_type != '{}|{}'.format(UNKNOWN_EVENT_KEY, UNKNOWN_EVENT_KEY):
            #     # ****** lookup sentence in corpus *********
            #     #sentence = self.output_corpus.eid2sent[ori_example['entry_id']]
            #     sentence = self.output_corpus.eid2sent[ori_example.sentence.docid]
            #     # ****** add annotations to sentence *******
            #     event_id = f'event{len(sentence.abstract_events) + 1}'
            #     anchor_ss_id = sentence.add_span_set(span_strings=[anchor_word])
            #     anchor_span_set = sentence.span_sets[anchor_ss_id]
            #     agent_span_sets, patient_span_sets = [], []
            #     agent_ss_id_list = []
            #     patient_ss_id_list = []
            #
            #     output_agent_offsets = {}
            #     output_patient_offsets = {}
            #
            #     if len(agents) > 0:
            #         for agent in agents:
            #             ss_id = sentence.add_span_set(
            #                 span_strings=[agent])
            #             agent_ss_id_list.append(ss_id)
            #             output_agent_offsets[ss_id] = [agent]
            #
            #     if len(patients) > 0:
            #         for patient in patients:
            #             ss_id = sentence.add_span_set(
            #                 span_strings=[patient])
            #             patient_ss_id_list.append(ss_id)
            #             output_patient_offsets[ss_id] = [patient]
            #
            #     for ss_id in agent_ss_id_list:
            #         agent_span_sets.append(sentence.span_sets[ss_id])
            #     for ss_id in patient_ss_id_list:
            #         patient_span_sets.append(sentence.span_sets[ss_id])
            #     abstract_event = AbstractEvent(
            #         event_id=event_id,
            #         #helpful_harmful=event_type.split('|')[0],	# <==
            #         #material_verbal=event_type.split('|')[1],	# <==
            #         helpful_harmful=event_type.split('.')[0],	# ==>
            #         material_verbal=event_type.split('.')[1],	# ==>
            #         anchor_span_set=anchor_span_set,
            #         agent_span_sets=agent_span_sets,
            #         patient_span_sets=patient_span_sets,
            #         anchor_offsets={
            #             anchor_ss_id: anchor_offset
            #         },
            #         agent_offsets=output_agent_offsets,
            #         patient_offsets=output_patient_offsets
            #     )
            #     sentence.add_abstract_event(abstract_event=abstract_event)

    def get_new_xlmr_ids(self, eid, trigger_tok):
        example = self.test_iterator.id2ori_example[eid]
        #word_list = example['word']
        word_list = example.words
        
        trigger_toks = [trigger_tok]
        #trigger_word = example['text'][
        #               example['span'][int(trigger_toks[0])][0]: example['span'][int(trigger_toks[-1])][1] + 1]
        #print('trigger_tok=', trigger_tok)
        #print('len(word_list)=', len(word_list))
        #print(word_list)
        trigger_word = word_list[trigger_tok]

        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.trainer['argument'].model.xlmr_embedding, word_list,
                                                              trigger_word)
        return xlmr_ids, trigger_word

    def get_inputs_for_argument_model(self, old_inputs, ED_preds, ED_probs):
        new_inputs = {
            'xlmr_ids': [],
            'biw2v_ids': [],
            'retrieve_ids': [],
            'upos_ids': [],
            'xpos_ids': [],
            'head_ids': [],
            'deprel_ids': [],
            'ner_ids': [],
            'triggers': [],
            'eid': [],
            'pad_masks': []
        }
        xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, eid, pad_masks = old_inputs
        eid = eid.data.cpu().numpy()
        batch_size, seq_len = ED_preds.shape
        ED_preds = ED_preds.data.cpu().numpy()
        ED_probs = ED_probs.data.cpu().numpy()

        #print('ED_preds:', ED_preds)
        #print('batch_size=', batch_size, 'seq_len=', seq_len)

        trigger_words = []
        event_types = []
        event_probs = []

        for b_id in range(batch_size):

            my_seq_len = len(self.test_iterator.id2ori_example[eid[b_id]].words)

            #print('labels=', self.test_iterator.id2ori_example[eid[b_id]].labels)
            ed_labels = self.test_iterator.id2ori_example[eid[b_id]].labels
            if self.opt['decode_with_gold_triggers']:
                trigger_toks = [k for k in range(len(ed_labels)) if ed_labels[k] != 'O']
            else:
                trigger_toks = [k for k in range(my_seq_len) if ED_preds[b_id][k] != self.trigger_label_map['O']]
            #print('trigger_toks=', trigger_toks)

            #if np.sum(ED_preds[b_id]) > 0:  # we cannot do this because our trigger label map is different
            if len(trigger_toks) > 0:
                #trigger_toks = [k for k in range(seq_len) if ED_preds[b_id][k] > 0] # we cannot do this because our trigger label map is different
                for trigger_tok in trigger_toks:
                    brt_ids, trigger_word = self.get_new_xlmr_ids(eid[b_id], trigger_tok)
                    if brt_ids is None:
                        continue
                    trigger_words.append([trigger_tok])
                    if self.opt['decode_with_gold_triggers']:
                        event_types.append(ed_labels[trigger_tok])
                    else:
                        event_types.append(self.event_inverse_map[ED_preds[b_id][trigger_tok]])
                    event_probs.append(ED_probs[b_id][trigger_tok])

                    new_inputs['xlmr_ids'].append(brt_ids)
                    new_inputs['biw2v_ids'].append(biw2v_ids[b_id])
                    new_inputs['retrieve_ids'].append(retrieve_ids[b_id])
                    new_inputs['upos_ids'].append(upos_ids[b_id])
                    new_inputs['xpos_ids'].append(xpos_ids[b_id])
                    new_inputs['head_ids'].append(head_ids[b_id])
                    new_inputs['deprel_ids'].append(deprel_ids[b_id])
                    new_inputs['ner_ids'].append(ner_ids[b_id])
                    new_inputs['triggers'].append(trigger_tok)
                    new_inputs['eid'].append(eid[b_id])
                    new_inputs['pad_masks'].append(pad_masks[b_id])

        if len(new_inputs['xlmr_ids']) == 0:
            return None, None, None, None, None
        else:
            new_inputs['xlmr_ids'] = do_padding(new_inputs['xlmr_ids'], len(new_inputs['xlmr_ids'])).long()
            new_inputs['biw2v_ids'] = torch.stack(new_inputs['biw2v_ids'], dim=0).long()
            new_inputs['retrieve_ids'] = torch.stack(new_inputs['retrieve_ids'], dim=0).long()
            new_inputs['upos_ids'] = torch.stack(new_inputs['upos_ids'], dim=0).long()
            new_inputs['xpos_ids'] = torch.stack(new_inputs['xpos_ids'], dim=0).long()
            new_inputs['head_ids'] = torch.stack(new_inputs['head_ids'], dim=0).long()
            new_inputs['deprel_ids'] = torch.stack(new_inputs['deprel_ids'], dim=0).long()
            new_inputs['ner_ids'] = torch.stack(new_inputs['ner_ids'], dim=0).long()
            new_inputs['triggers'] = torch.Tensor(new_inputs['triggers']).long()
            new_inputs['eid'] = torch.Tensor(new_inputs['eid']).long()
            new_inputs['pad_masks'] = torch.stack(new_inputs['pad_masks'], dim=0).long()

            with torch.no_grad():
                max_seq_len = torch.max(torch.sum(new_inputs['pad_masks'].eq(0), 1))

            new_inputs['biw2v_ids'] = new_inputs['biw2v_ids'][:, :max_seq_len]
            new_inputs['retrieve_ids'] = new_inputs['retrieve_ids'][:, :max_seq_len]
            new_inputs['upos_ids'] = new_inputs['upos_ids'][:, :max_seq_len]
            new_inputs['xpos_ids'] = new_inputs['xpos_ids'][:, :max_seq_len]
            new_inputs['head_ids'] = new_inputs['head_ids'][:, :max_seq_len]
            new_inputs['deprel_ids'] = new_inputs['deprel_ids'][:, :max_seq_len]
            new_inputs['ner_ids'] = new_inputs['ner_ids'][:, :max_seq_len]
            new_inputs['pad_masks'] = new_inputs['pad_masks'][:, :max_seq_len]

            return (new_inputs['xlmr_ids'].to(self.opt['device']),
                    new_inputs['biw2v_ids'].to(self.opt['device']),
                    new_inputs['retrieve_ids'].to(self.opt['device']),
                    new_inputs['upos_ids'].to(self.opt['device']),
                    new_inputs['xpos_ids'].to(self.opt['device']),
                    new_inputs['head_ids'].to(self.opt['device']),
                    new_inputs['deprel_ids'].to(self.opt['device']),
                    new_inputs['ner_ids'].to(self.opt['device']),
                    new_inputs['triggers'].to(self.opt['device']),
                    new_inputs['eid'].to(self.opt['device']),
                    new_inputs['pad_masks'].to(self.opt['device'])), new_inputs['eid'].to(
                self.opt['device']), trigger_words, event_types, event_probs
