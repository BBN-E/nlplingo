from .ED_trainer import EDTrainer
from .argument_trainer import ArgumentTrainer
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.iterators import PipelineIterator, get_arguments
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.local_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *
import torch
from nlplingo.oregon.event_models.uoregon.define_opt import logger
from datetime import datetime
from nlplingo.oregon.event_models.uoregon.tools.corpus_utils import Corpus, LANG_SET, AbstractEvent, get_ori_string, \
    strip_punctuations
from nlplingo.oregon.event_models.uoregon.tools.xlmr import xlmr_tokenizer


class PipelineTrainer:
    def __init__(self, opt, test_file, lang):
        assert lang in LANG_SET, 'Unsupported language! Select from: {}'.format(LANG_SET)
        self.opt = opt
        self.lang = lang

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
            'ED': EDTrainer(config['ED'], eval_mode=True),                  # TODO call the one in tasks.sequence
            'argument': ArgumentTrainer(config['argument'], eval_mode=True) # TODO call the one in tasks.sequence
        }

        self.trainer['ED'].model.eval()
        self.trainer['argument'].model.eval()

        # load the stanford parsed examples from the JSON file, encode them, create batches out of them
        self.test_iterator = PipelineIterator(
            xlmr_model=self.trainer['ED'].model.xlmr_embedding,
            # data_path=os.path.join(MODEL_DIR,			# <==
            data_path=os.path.join(opt['datapoint_dir'],	# ==>
                                   opt['data_map']['pipeline']['test'].format(get_data_dir(self.opt['data'])))
        )
        print('Pipeline : Test data: {}'.format(self.test_iterator.num_examples))

        print('Ingesting test_file=', test_file)
        self.output_corpus = Corpus(filepath=test_file, lang=lang, parsing=False)
        self.output_corpus.clear_annotation()

        # TODO below
        self.impact_inverse_map = dict([(v, k) for k, v in IMPACT_MAP.items()])     # not used anywhere else
        self.effect_inverse_map = dict([(v, k) for k, v in EFFECT_MAP.items()])     # not used anywhere else
        self.event_inverse_map = dict([(v, k) for k, v in EVENT_MAP.items()])

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

    def load_ckpt_configs(self):
        # When you specify a particular ED_eval_epoch and a particular argument_eval_epoch, this code will try to find
        # that for you. If cannot find, will regress to the first checkpoint it can find.
        ED_config_path = os.path.join('checkpoints',
                                      'ED.epoch-{}.config'.format(self.opt['ED_eval_epoch']))
        argument_config_path = os.path.join('checkpoints',
                                            'argument.epoch-{}.config'.format(self.opt['argument_eval_epoch']))

        if not os.path.exists(ED_config_path) or not os.path.join(argument_config_path):
            ED_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
                               len(re.findall(
                                   r'ED[.]epoch[-]\d+[.]config',
                                   fname)) > 0]
            argument_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
                                     len(re.findall(
                                         r'argument[.]epoch[-]\d+[.]config',
                                         fname)) > 0]
            if len(ED_config_paths) * len(argument_config_paths) == 0:
                print('Not enough checkpoints!')
                if self.opt['docker_run'] == 0:
                    print('You might have not renamed checkpoint files from * -> *.hideval.simulated')
                exit(1)
            else:
                print('{} and {} are not found!\nUsing {} and {} instead!'.format(
                    ED_config_path,
                    argument_config_path,
                    ED_config_paths[0],
                    argument_config_paths[0]
                ))
                ED_config_path = ED_config_paths[0]
                argument_config_path = argument_config_paths[0]
                self.opt['ED_eval_epoch'] = int(
                    re.findall(r'epoch[-](\d+)[.]config',
                               ED_config_path)[0])
                self.opt['argument_eval_epoch'] = int(
                    re.findall(r'epoch[-](\d+)[.]config',
                               argument_config_path)[0])
        # Just ensure that by now, ED_config_path points to something like: ED.epoch-4.config
        # Just ensure that by now, argument_config_path points to something like: argument.epoch-4.config

        ED_config = read_json(ED_config_path)
        argument_config = read_json(argument_config_path)
        for param in ['biw2v_vecs', 'device', 'ED_eval_epoch', 'argument_eval_epoch', 'train_on_arb']:
            ED_config[param] = self.opt[param]
            argument_config[param] = self.opt[param]
        for param in self.opt:
            if param not in ED_config:
                ED_config[param] = self.opt[param]
            if param not in argument_config:
                argument_config[param] = self.opt[param]
        return {'ED': ED_config, 'argument': argument_config}

    def eval(self):
        self.trainer['ED'].load_saved_model()
        self.trainer['argument'].load_saved_model()

        for b_id, inputs in enumerate(self.test_iterator):
            inputs = move_to_device(inputs, self.opt['device'])

            ED_preds, ED_probs = self.trainer['ED'].predict(inputs)

            argument_inputs, eids, trigger_words, event_types, event_probs = self.get_inputs_for_argument_model(inputs,
                                                                                                                ED_preds,
                                                                                                                ED_probs)
            """
            eids= tensor([1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 5, 5, 2, 2, 2, 2, 9, 9, 9, 9, 7, 7, 8, 8,
                    4, 4, 4, 4, 0, 0], device='cuda:0')
            trigger_words= [[1], [22], [24], [2], [3], [11], [23], [4], [12], [25], [0], [16], [1], [8], [10], [17], [3], [7], [8], [11], [2], [10], [0], [6], [0], [8], [13], [15], [0], [18]]
            event_types= ['neutral|verbal', 'neutral|both', 'helpful|material', 'helpful|verbal', 'neutral|verbal', 'neutral|both', 'helpful|material', 'neutral|material', 'harmful|material', 'harmful|material', 'neutral|verbal', 'harmful|both', 'neutral|verbal', 'neutral|verbal', 'harmful|material', 'neutral|verbal', 'harmful|material', 'harmful|material', 'harmful|material', 'neutral|verbal', 'neutral|material', 'harmful|material', 'neutral|verbal', 'neutral|verbal', 'neutral|material', 'neutral|material', 'harmful|both', 'harmful|material', 'neutral|verbal', 'neutral|verbal']
            event_probs= [array([4.0915366e-03, 8.9600036e-07, 2.5587494e-07, 7.9609504e-07,
                   2.4147303e-05, 4.4063483e-05, 4.8672303e-04, 8.5377338e-05,
                   1.3467354e-05, 4.6803943e-05, 9.3863020e-04, 9.7463642e-05,
                   1.9753193e-04, 3.6581000e-04, 9.9335068e-01, 2.5578309e-04],
                  dtype=float32), array([8.2305036e-03, 1.4697574e-05, 6.1192168e-06, 2.0656869e-05,
                   1.7769105e-04, 2.2180121e-04, 3.3155660e-04, 2.5483756e-03,
                   4.3701223e-04, 1.4507882e-02, 2.3916408e-03, 7.4849330e-02,
                   2.9614938e-03, 2.2396229e-02, 1.7600251e-02, 8.5330486e-01],
                  dtype=float32), array([2.0966575e-01, 5.6851190e-05, 2.1032634e-05, 5.5651773e-05,
                   1.7963679e-03, 3.0679714e-02, 6.1572284e-02, 8.8818245e-02,
                   5.4130296e-04, 3.1282967e-01, 9.9438727e-03, 7.9179555e-02,
                   6.6516870e-03, 3.4592703e-02, 1.0611436e-02, 1.5298392e-01],
                  dtype=float32), array([1.9942187e-01, 1.2716409e-05, 7.3963520e-06, 1.9074489e-05,
                   2.0656030e-04, 1.8359447e-04, 3.3479647e-03, 1.1254769e-04,
                   3.1872917e-04, 1.4747353e-01, 5.5890936e-01, 1.4941645e-02,
                   8.0368007e-03, 3.7771977e-02, 2.7622486e-02, 1.6137511e-03],
                  dtype=float32), array([5.9900293e-03, 2.5699912e-05, 1.0599404e-05, 2.9877285e-05,
                   4.5747572e-04, 2.6965313e-04, 3.9312471e-02, 1.3108774e-03,
                   1.8380856e-04, 3.0923134e-04, 1.8477011e-02, 2.3279074e-03,
                   3.1589132e-03, 1.7323957e-03, 9.2243010e-01, 3.9738868e-03],
                  dtype=float32), array([6.9429474e-03, 2.4729659e-05, 1.0984882e-05, 2.8686743e-05,
                   5.7426828e-04, 5.1303388e-04, 3.5867940e-03, 1.1702468e-02,
                   3.6871288e-04, 3.2067860e-03, 7.6875463e-03, 5.0800730e-02,
                   1.6110454e-02, 2.7652135e-02, 2.2584496e-01, 6.4494479e-01],
                  dtype=float32), array([4.1497463e-01, 1.1148420e-05, 7.5135167e-06, 1.5786940e-05,
                   2.4018195e-04, 4.7952464e-04, 9.7997533e-04, 2.5670091e-04,
                   3.4777072e-04, 4.2628238e-01, 7.4442782e-02, 2.3955498e-02,
                   8.9166751e-03, 3.8109951e-02, 3.6447800e-03, 7.3348205e-03],
                  dtype=float32), array([9.4591489e-04, 5.3036274e-06, 1.0563388e-05, 5.8668529e-06,
                   4.6770507e-04, 5.2893169e-02, 3.0739306e-04, 1.5054536e-03,
                   9.9412537e-05, 6.7200579e-02, 4.2583654e-04, 1.9927460e-03,
                   7.8673372e-03, 8.5320592e-01, 2.8677678e-03, 1.0199070e-02],
                  dtype=float32), array([7.9694197e-02, 2.1066339e-06, 1.3104986e-06, 9.0103867e-07,
                   5.1638269e-04, 9.0158814e-01, 9.9835533e-04, 2.9750168e-03,
                   6.4270862e-06, 1.6248672e-03, 4.1470572e-05, 3.6083489e-05,
                   1.6672844e-03, 1.0301739e-02, 4.4709505e-04, 9.8589735e-05],
                  dtype=float32), array([6.59529399e-03, 8.23955816e-06, 1.29269665e-05, 7.84954864e-06,
                   1.37526169e-03, 7.91635692e-01, 6.22351130e-04, 3.56035470e-03,
                   6.28239213e-05, 2.01822221e-02, 2.20770817e-04, 3.24975525e-04,
                   1.32688116e-02, 1.58275321e-01, 1.97326369e-03, 1.87385851e-03],
                  dtype=float32), array([2.0243383e-03, 1.4849456e-06, 5.2986366e-07, 1.3255053e-06,
                   3.2956232e-05, 4.0225535e-05, 5.6687568e-04, 9.8343153e-05,
                   1.8798533e-05, 4.0064315e-05, 1.2114695e-03, 1.3698499e-04,
                   3.4574312e-04, 4.2817165e-04, 9.9471855e-01, 3.3398462e-04],
                  dtype=float32), array([2.0776312e-01, 2.1074657e-05, 5.3606796e-06, 6.9603630e-06,
                   1.3591538e-03, 4.8153087e-02, 5.0951805e-02, 6.0072887e-01,
                   4.0763571e-05, 2.3762751e-03, 3.3022446e-04, 9.8846089e-03,
                   8.9644594e-03, 9.7484253e-03, 2.5205018e-03, 5.7145331e-02],
                  dtype=float32), array([4.09231475e-03, 9.55663836e-07, 3.29764049e-07, 8.68083816e-07,
                   3.04486730e-05, 4.74395856e-05, 8.39301327e-04, 5.28332275e-05,
                   1.29687787e-05, 4.68769722e-05, 2.10635108e-03, 9.31487593e-05,
                   4.14624519e-04, 6.61705853e-04, 9.91492689e-01, 1.07074506e-04],
                  dtype=float32), array([1.4907821e-01, 3.0793039e-05, 1.0054615e-05, 4.0907009e-05,
                   4.4189012e-04, 4.8633051e-04, 8.2699142e-02, 3.0247862e-03,
                   2.2250887e-04, 2.0464871e-03, 2.5921375e-01, 1.4450299e-02,
                   6.5038283e-03, 1.5005728e-03, 4.7679240e-01, 3.4581437e-03],
                  dtype=float32), array([3.3068703e-03, 2.3949510e-06, 1.7240760e-06, 8.3340342e-07,
                   4.5662367e-04, 9.8403972e-01, 1.4203453e-03, 4.9549611e-03,
                   4.2958909e-06, 1.0668292e-03, 2.1354783e-05, 4.0414958e-05,
                   6.7935034e-04, 3.5062162e-03, 4.0625563e-04, 9.1869122e-05],
                  dtype=float32), array([1.0846975e-03, 7.2659218e-06, 4.0844234e-06, 1.3355348e-05,
                   1.3569294e-04, 2.5317702e-04, 8.9161042e-03, 4.5850253e-04,
                   1.1089002e-04, 9.2073152e-04, 4.7915500e-02, 2.0165534e-03,
                   2.2478167e-03, 2.0917456e-03, 9.3187588e-01, 1.9480775e-03],
                  dtype=float32), array([3.8340285e-02, 2.1884227e-05, 9.0778603e-06, 7.8481526e-06,
                   2.1743868e-03, 5.8083659e-01, 3.2065985e-01, 4.5555484e-02,
                   1.8858513e-05, 6.4633851e-04, 3.7270033e-04, 5.2951701e-04,
                   2.3308329e-03, 3.3545385e-03, 4.6323300e-03, 5.0943240e-04],
                  dtype=float32), array([1.5988480e-01, 5.9191993e-06, 2.7143963e-06, 1.8595219e-06,
                   1.2252619e-03, 7.9838133e-01, 2.4132940e-03, 1.0464846e-02,
                   1.3555631e-05, 1.5584956e-03, 6.1179111e-05, 1.1397924e-04,
                   2.6227681e-03, 2.2273807e-02, 7.1842934e-04, 2.5782094e-04],
                  dtype=float32), array([9.8272063e-02, 6.4103797e-06, 2.1908781e-06, 1.5253295e-06,
                   8.5878751e-04, 8.7532866e-01, 1.6443047e-03, 1.7656397e-02,
                   8.1387807e-06, 3.8723962e-04, 2.9603892e-05, 6.0838262e-05,
                   1.1115507e-03, 3.9970917e-03, 4.1978617e-04, 2.1545788e-04],
                  dtype=float32), array([2.1012580e-02, 2.7796705e-05, 1.2643658e-05, 3.0387506e-05,
                   1.1067863e-03, 6.2150313e-03, 4.3626249e-02, 3.1704821e-02,
                   2.1489309e-04, 3.4670150e-03, 1.3327037e-02, 1.5880786e-02,
                   1.9373838e-02, 6.7254782e-02, 7.1293718e-01, 6.3808121e-02],
                  dtype=float32), array([6.2678368e-03, 1.8630730e-06, 2.0766131e-06, 1.9129891e-06,
                   1.1024540e-04, 1.5553567e-03, 7.2017727e-05, 2.1591465e-04,
                   6.1476901e-05, 2.2231951e-02, 2.5876373e-04, 1.1510118e-03,
                   2.4884483e-03, 9.5846933e-01, 1.5675240e-03, 5.5442834e-03],
                  dtype=float32), array([3.6830388e-03, 1.8682682e-06, 1.2555945e-06, 6.9780924e-07,
                   3.8756614e-04, 9.8167038e-01, 5.5378751e-04, 6.0314708e-03,
                   3.9861020e-06, 4.2377083e-04, 1.8810753e-05, 2.7387056e-05,
                   7.0670788e-04, 5.7942970e-03, 5.0450535e-04, 1.9045352e-04],
                  dtype=float32), array([3.7560479e-03, 1.7332015e-06, 7.1110986e-07, 2.2672787e-06,
                   4.2984193e-05, 4.8701804e-05, 8.5468910e-04, 7.6431279e-05,
                   2.8530285e-05, 7.0654089e-05, 2.4900865e-03, 1.8743023e-04,
                   6.5291772e-04, 6.5630849e-04, 9.9066621e-01, 4.6436946e-04],
                  dtype=float32), array([2.33248025e-01, 1.00022480e-05, 2.63102879e-06, 1.23083028e-05,
                   2.12651270e-04, 1.15277384e-04, 2.07207515e-03, 6.18761696e-04,
                   1.67175225e-04, 3.43918131e-04, 6.30066125e-03, 2.94535537e-03,
                   4.54532774e-03, 2.71108653e-03, 7.33279765e-01, 1.34149976e-02],
                  dtype=float32), array([1.2495068e-01, 3.4042125e-06, 1.7341179e-06, 2.4570327e-06,
                   2.0163966e-04, 1.9129541e-03, 3.4798685e-04, 8.7704713e-04,
                   7.8387297e-05, 4.3740398e-03, 9.1194169e-04, 1.3165748e-03,
                   3.4687372e-03, 8.2813710e-01, 2.5806554e-02, 7.6088365e-03],
                  dtype=float32), array([3.6627325e-01, 5.8573341e-06, 2.1352284e-06, 2.8719926e-06,
                   3.8143055e-04, 2.9455153e-03, 2.8080100e-04, 5.9475461e-03,
                   7.6334654e-05, 2.2175133e-03, 2.4744257e-04, 2.2156371e-03,
                   5.8619892e-03, 5.6800508e-01, 1.0214765e-02, 3.5321854e-02],
                  dtype=float32), array([3.0611081e-02, 1.9100331e-05, 3.2082216e-06, 6.0098309e-06,
                   4.0594381e-04, 2.3363738e-03, 2.1089401e-02, 8.9009708e-01,
                   1.8916444e-05, 1.4113207e-04, 4.0131848e-04, 4.7117192e-03,
                   1.5661733e-03, 6.2861515e-04, 5.8666090e-03, 4.2097267e-02],
                  dtype=float32), array([3.7127376e-02, 1.9383672e-06, 1.0730172e-06, 6.4203067e-07,
                   3.1931041e-04, 9.4751894e-01, 6.9519121e-04, 5.0136419e-03,
                   4.8033098e-06, 1.0177845e-03, 4.5260684e-05, 2.6248892e-05,
                   5.8338157e-04, 7.1903169e-03, 3.5604328e-04, 9.7902746e-05],
                  dtype=float32), array([2.9217969e-03, 1.2115049e-06, 4.4519584e-07, 1.3202434e-06,
                   2.9414625e-05, 5.7706598e-05, 5.1405560e-04, 6.1732215e-05,
                   1.8563800e-05, 5.3108150e-05, 1.6366981e-03, 9.5327283e-05,
                   3.0477601e-04, 7.4743625e-04, 9.9328071e-01, 2.7579395e-04],
                  dtype=float32), array([3.86206806e-02, 1.47527135e-05, 3.82810822e-06, 1.42902318e-05,
                   2.72676640e-04, 3.13856814e-04, 9.18926962e-04, 3.10462038e-03,
                   2.74572929e-04, 5.81537606e-04, 4.07564174e-03, 5.05779451e-03,
                   5.00261039e-03, 1.46071417e-02, 7.12119818e-01, 2.15017214e-01],
                  dtype=float32)]
            """

            if argument_inputs is not None:
                argument_preds, argument_probs = self.trainer['argument'].predict(argument_inputs)
                self.record_predictions(eids, trigger_words, event_types, event_probs, argument_preds, argument_probs)

        print('Writing predictions to file: {}'.format(self.opt['output_file']))
        self.output_corpus.save(output_file=self.opt['output_file'])

        logger.info('Writing predictions to file: {}'.format(self.opt['output_file']))

    def record_predictions(self, eids, trigger_toks_list, event_types, event_probs, argument_preds, argument_probs):
        eids = eids.data.cpu().numpy()
        for k, eid in enumerate(eids):
            ori_example = self.test_iterator.id2ori_example[eid]
            trigger_toks = trigger_toks_list[k]
            arg_preds = argument_preds[k]
            ###################### extract texts ########################
            _, _, agent_norm_offsets, patient_norm_offsets = get_arguments(self.trainer['argument'].id2tag,
                                                                           arg_preds, ori_example)
            ############## recover original strings and offsets ##########
            ############## OFFSETS ARE OPEN INTERVALS ####################
            anchor_norm_offset = [
                ori_example['span'][int(trigger_toks[0])][0], ori_example['span'][int(trigger_toks[-1])][
                                                                  1] + 1]  # (x, y + 1)
            anchor_offset, anchor_word = get_ori_string(ori_example, norm_offset=anchor_norm_offset)
            # -------------------------------
            agents, patients = [], []
            agent_offsets, patient_offsets = {}, {}
            for agent in agent_norm_offsets:
                agent_norm_offset = agent_norm_offsets[agent]
                agent_ori_offset, agent_ori_string = get_ori_string(ori_example, norm_offset=agent_norm_offset)
                if len(agent_ori_string) > 0:
                    agents.append(agent_ori_string)
                    agent_offsets[agent_ori_string] = agent_ori_offset

            for patient in patient_norm_offsets:
                patient_norm_offset = patient_norm_offsets[patient]
                patient_ori_offset, patient_ori_string = get_ori_string(ori_example, norm_offset=patient_norm_offset)
                if len(patient_ori_string) > 0:
                    patients.append(patient_ori_string)
                    patient_offsets[patient_ori_string] = patient_ori_offset
            ###############################################################
            agents = list(set(agents))
            patients = list(set(patients))

            agents = [agent for agent in agents if len(agent.strip()) > 0]
            patients = [patient for patient in patients if len(patient.strip()) > 0]

            event_type = event_types[k]
            event_prob = event_probs[k]
            argument_prob = argument_probs[k]
            if event_type != '{}|{}'.format(UNKNOWN_EVENT_KEY, UNKNOWN_EVENT_KEY):
                # ****** lookup sentence in corpus *********
                sentence = self.output_corpus.eid2sent[ori_example['entry_id']]
                # ****** add annotations to sentence *******
                event_id = f'event{len(sentence.abstract_events) + 1}'
                anchor_ss_id = sentence.add_span_set(span_strings=[anchor_word])
                anchor_span_set = sentence.span_sets[anchor_ss_id]
                agent_span_sets, patient_span_sets = [], []
                agent_ss_id_list = []
                patient_ss_id_list = []

                output_agent_offsets = {}
                output_patient_offsets = {}

                if len(agents) > 0:
                    for agent in agents:
                        ss_id = sentence.add_span_set(
                            span_strings=[agent])
                        agent_ss_id_list.append(ss_id)
                        output_agent_offsets[ss_id] = [agent]

                if len(patients) > 0:
                    for patient in patients:
                        ss_id = sentence.add_span_set(
                            span_strings=[patient])
                        patient_ss_id_list.append(ss_id)
                        output_patient_offsets[ss_id] = [patient]

                for ss_id in agent_ss_id_list:
                    agent_span_sets.append(sentence.span_sets[ss_id])
                for ss_id in patient_ss_id_list:
                    patient_span_sets.append(sentence.span_sets[ss_id])
                abstract_event = AbstractEvent(
                    event_id=event_id,
                    helpful_harmful=event_type.split('|')[0],
                    material_verbal=event_type.split('|')[1],
                    anchor_span_set=anchor_span_set,
                    agent_span_sets=agent_span_sets,
                    patient_span_sets=patient_span_sets,
                    anchor_offsets={
                        anchor_ss_id: anchor_offset
                    },
                    agent_offsets=output_agent_offsets,
                    patient_offsets=output_patient_offsets
                )
                sentence.add_abstract_event(abstract_event=abstract_event)

    def get_new_xlmr_ids(self, eid, trigger_tok):
        example = self.test_iterator.id2ori_example[eid]
        word_list = example['word']
        trigger_toks = [trigger_tok]
        trigger_word = example['text'][
                       example['span'][int(trigger_toks[0])][0]: example['span'][int(trigger_toks[-1])][1] + 1]
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

        """
        xlmr_ids.shape= torch.Size([10, 53])
        biw2v_ids.shape= torch.Size([10, 33])
        retrieve_ids.shape= torch.Size([10, 33])
        upos_ids.shape= torch.Size([10, 33])
        xpos_ids.shape= torch.Size([10, 33])
        head_ids.shape= torch.Size([10, 33])
        deprel_ids.shape= torch.Size([10, 33])
        ner_ids.shape= torch.Size([10, 33])
        eid.shape= (10,)
        pad_masks.shape= torch.Size([10, 33])
        batch_size= 10
        seq_len= 33
        ED_preds.shape= (10, 33)
        ED_probs.shape= (10, 33, 16)
        """

        trigger_words = []
        event_types = []
        event_probs = []

        for b_id in range(batch_size):
            if np.sum(ED_preds[b_id]) > 0:
                trigger_toks = [k for k in range(seq_len) if ED_preds[b_id][k] > 0]

                """
                b_id= 0
                ED_preds[b_id]= [ 0 14  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 15  0
                  9  0  0  0  0  0  0  0  0]
                trigger_toks= [1, 22, 24]
                """

                for trigger_tok in trigger_toks:
                    brt_ids, trigger_word = self.get_new_xlmr_ids(eid[b_id], trigger_tok)
                    if brt_ids is None:
                        continue
                    trigger_words.append([trigger_tok])
                    event_types.append(self.event_inverse_map[ED_preds[b_id][trigger_tok]])
                    event_probs.append(ED_probs[b_id][trigger_tok])

                    """
                    trigger_tok= 1
                    brt_ids= tensor([     0,      6,      5,  90621,  47229,    250,    181,   5273,  10408,
                              6267,   4039,  31245,  71633,   2620,  18684,   6466,   7233,    250,
                               240, 102468,    368,      6, 185701,  35618,  18004, 159565,  97288,
                             41468,    152,     94,  13231,   3108,    746,  14272,   3070, 102935,
                              2103, 153872,    767, 186386,  12581,  30039,    230,  59721, 148726,
                               755,    230,   6816,   1692,    340,      6,      5,      2,      2,
                             90621,      2])
                    trigger_word= قالت
                    self.event_inverse_map[ED_preds[b_id][trigger_tok]]= neutral|verbal
                    ED_probs[b_id][trigger_tok]= [4.0915366e-03 8.9600036e-07 2.5587494e-07 7.9609504e-07 2.4147303e-05
                     4.4063483e-05 4.8672303e-04 8.5377338e-05 1.3467354e-05 4.6803943e-05
                     9.3863020e-04 9.7463642e-05 1.9753193e-04 3.6581000e-04 9.9335068e-01
                     2.5578309e-04]
                     
                    trigger_tok= 22
                    brt_ids= tensor([     0,      6,      5,  90621,  47229,    250,    181,   5273,  10408,
                              6267,   4039,  31245,  71633,   2620,  18684,   6466,   7233,    250,
                               240, 102468,    368,      6, 185701,  35618,  18004, 159565,  97288,
                             41468,    152,     94,  13231,   3108,    746,  14272,   3070, 102935,
                              2103, 153872,    767, 186386,  12581,  30039,    230,  59721, 148726,
                               755,    230,   6816,   1692,    340,      6,      5,      2,      2,
                            102935,      2])
                    trigger_word= الانتخابات
                    self.event_inverse_map[ED_preds[b_id][trigger_tok]]= neutral|both
                    ED_probs[b_id][trigger_tok]= [8.2305036e-03 1.4697574e-05 6.1192168e-06 2.0656869e-05 1.7769105e-04
                     2.2180121e-04 3.3155660e-04 2.5483756e-03 4.3701223e-04 1.4507882e-02
                     2.3916408e-03 7.4849330e-02 2.9614938e-03 2.2396229e-02 1.7600251e-02
                     8.5330486e-01]
                    
                    trigger_tok= 24
                    brt_ids= tensor([     0,      6,      5,  90621,  47229,    250,    181,   5273,  10408,
                              6267,   4039,  31245,  71633,   2620,  18684,   6466,   7233,    250,
                               240, 102468,    368,      6, 185701,  35618,  18004, 159565,  97288,
                             41468,    152,     94,  13231,   3108,    746,  14272,   3070, 102935,
                              2103, 153872,    767, 186386,  12581,  30039,    230,  59721, 148726,
                               755,    230,   6816,   1692,    340,      6,      5,      2,      2,
                               767, 186386,      2])
                    trigger_word= يحرم
                    self.event_inverse_map[ED_preds[b_id][trigger_tok]]= helpful|material
                    ED_probs[b_id][trigger_tok]= [2.0966575e-01 5.6851190e-05 2.1032634e-05 5.5651773e-05 1.7963679e-03
                     3.0679714e-02 6.1572284e-02 8.8818245e-02 5.4130296e-04 3.1282967e-01
                     9.9438727e-03 7.9179555e-02 6.6516870e-03 3.4592703e-02 1.0611436e-02
                     1.5298392e-01]
                    """

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

            """
            new_inputs['xlmr_ids'].shape= torch.Size([30, 57])
            new_inputs['biw2v_ids'].shape= torch.Size([30, 33])
            new_intpus['retrieve_ids'].shape= torch.Size([30, 33])
            new_inputs['upos_ids'].shape= torch.Size([30, 33])
            new_inputs['xpos_ids'].shape= torch.Size([30, 33])
            new_inputs['head_ids'].shape= torch.Size([30, 33])
            new_inputs['deprel_ids'].shape= torch.Size([30, 33])
            new_inputs['ner_ids'].shape= torch.Size([30, 33])
            new_inputs['trigger'].shape= torch.Size([30])
            new_inputs['eid'].shape= torch.Size([30])
            new_inputs['pad_masks'].shape= torch.Size([30, 33])
            new_inputs['xlmr_ids']= tensor([[     0,      6,      5,  ...,  90621,      2,      0],
                    [     0,      6,      5,  ..., 102935,      2,      0],
                    [     0,      6,      5,  ...,    767, 186386,      2],
                    ...,
                    [     0, 105285,    368,  ...,      0,      0,      0],
                    [     0,   6625,  55468,  ...,      0,      0,      0],
                    [     0,   6625,  55468,  ...,      0,      0,      0]])
            new_inputs['biw2v_ids']= tensor([[     6, 114225, 113937, 128409,      1,   1443, 113675, 113666, 163097,
                     117713,      1, 126519,      1, 113266, 114068,      1,     50, 176957,
                          1, 113209, 127252, 113173, 113584, 120372, 126250, 113253, 113470,
                     113165, 117399, 113165, 119105, 177487,      6],
                    [     6, 114225, 113937, 128409,      1,   1443, 113675, 113666, 163097,
                     117713,      1, 126519,      1, 113266, 114068,      1,     50, 176957,
                          1, 113209, 127252, 113173, 113584, 120372, 126250, 113253, 113470,
                     113165, 117399, 113165, 119105, 177487,      6],
                    [     6, 114225, 113937, 128409,      1,   1443, 113675, 113666, 163097,
                     117713,      1, 126519,      1, 113266, 114068,      1,     50, 176957,
                          1, 113209, 127252, 113173, 113584, 120372, 126250, 113253, 113470,
                     113165, 117399, 113165, 119105, 177487,      6],
                    [     6, 113782,      1, 123638,      1,      1, 131450, 113546,      1,
                     116631, 113266, 114666,      1, 125284,      1, 115773, 117903, 124178,
                     113165, 113254,      1,    395, 113309,      1, 176957, 113203,      1,
                          1,      1, 177487,      6,      0,      0],
                    [     6, 113782,      1, 123638,      1,      1, 131450, 113546,      1,
                     116631, 113266, 114666,      1, 125284,      1, 115773, 117903, 124178,
                     113165, 113254,      1,    395, 113309,      1, 176957, 113203,      1,
                          1,      1, 177487,      6,      0,      0],
                    [     6, 113782,      1, 123638,      1,      1, 131450, 113546,      1,
                     116631, 113266, 114666,      1, 125284,      1, 115773, 117903, 124178,
                     113165, 113254,      1,    395, 113309,      1, 176957, 113203,      1,
                          1,      1, 177487,      6,      0,      0],
                    [     6, 113782,      1, 123638,      1,      1, 131450, 113546,      1,
                     116631, 113266, 114666,      1, 125284,      1, 115773, 117903, 124178,
                     113165, 113254,      1,    395, 113309,      1, 176957, 113203,      1,
                          1,      1, 177487,      6,      0,      0],
                    [113216, 113383,      1, 113448, 119129, 113264, 120182, 113167, 242005,
                     137590,      1, 113165, 137330,      1,      1, 234085, 179317, 115962,
                     162598, 114539, 114727, 114453,      1,      1, 114368, 114375,      6,
                          0,      0,      0,      0,      0,      0],
                    [113216, 113383,      1, 113448, 119129, 113264, 120182, 113167, 242005,
                     137590,      1, 113165, 137330,      1,      1, 234085, 179317, 115962,
                     162598, 114539, 114727, 114453,      1,      1, 114368, 114375,      6,
                          0,      0,      0,      0,      0,      0],
                    [113216, 113383,      1, 113448, 119129, 113264, 120182, 113167, 242005,
                     137590,      1, 113165, 137330,      1,      1, 234085, 179317, 115962,
                     162598, 114539, 114727, 114453,      1,      1, 114368, 114375,      6,
                          0,      0,      0,      0,      0,      0],
                    [113306, 115538, 113453,      1, 113183,      1, 120945,      1,      1,
                          1, 113209, 114243,      1, 113395,      1, 113216, 169807, 117207,
                     116183,      1, 192377, 121340,      6,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [113306, 115538, 113453,      1, 113183,      1, 120945,      1,      1,
                          1, 113209, 114243,      1, 113395,      1, 113216, 169807, 117207,
                     116183,      1, 192377, 121340,      6,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     6, 114821, 163342,      1, 130192,      1, 150092,      1, 191632,
                     113165,      1, 123445, 113381, 114054, 193520, 113200,      1, 113604,
                     134623, 113170, 122650,      6,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     6, 114821, 163342,      1, 130192,      1, 150092,      1, 191632,
                     113165,      1, 123445, 113381, 114054, 193520, 113200,      1, 113604,
                     134623, 113170, 122650,      6,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     6, 114821, 163342,      1, 130192,      1, 150092,      1, 191632,
                     113165,      1, 123445, 113381, 114054, 193520, 113200,      1, 113604,
                     134623, 113170, 122650,      6,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     6, 114821, 163342,      1, 130192,      1, 150092,      1, 191632,
                     113165,      1, 123445, 113381, 114054, 193520, 113200,      1, 113604,
                     134623, 113170, 122650,      6,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113621,      1, 123566,      1, 114004,      1,      1, 117675,
                     122110, 113447, 113985, 118076, 190493, 171192,      1, 113939,      1,
                     118621,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113621,      1, 123566,      1, 114004,      1,      1, 117675,
                     122110, 113447, 113985, 118076, 190493, 171192,      1, 113939,      1,
                     118621,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113621,      1, 123566,      1, 114004,      1,      1, 117675,
                     122110, 113447, 113985, 118076, 190493, 171192,      1, 113939,      1,
                     118621,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113621,      1, 123566,      1, 114004,      1,      1, 117675,
                     122110, 113447, 113985, 118076, 190493, 171192,      1, 113939,      1,
                     118621,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113381, 121106,      1, 118400, 113950, 113725, 113200,      1,
                     113165, 115868, 168565,      1, 117866, 113179,      1, 151891,      6,
                          0,      0,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [     1, 113381, 121106,      1, 118400, 113950, 113725, 113200,      1,
                     113165, 115868, 168565,      1, 117866, 113179,      1, 151891,      6,
                          0,      0,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [113306, 116827, 113219, 154925, 148370,      1, 113596,      1,      1,
                          1, 113219, 115695,      1, 114991, 113407, 156650, 115304, 113180,
                          1,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [113306, 116827, 113219, 154925, 148370,      1, 113596,      1,      1,
                          1, 113219, 115695,      1, 114991, 113407, 156650, 115304, 113180,
                          1,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [125488, 113398, 113165,      1, 113165,   9086, 116472,      1, 119791,
                     113604, 114109, 115046,      1, 123924, 117632, 119425, 113167, 120572,
                     113381,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [125488, 113398, 113165,      1, 113165,   9086, 116472,      1, 119791,
                     113604, 114109, 115046,      1, 123924, 117632, 119425, 113167, 120572,
                     113381,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [125488, 113398, 113165,      1, 113165,   9086, 116472,      1, 119791,
                     113604, 114109, 115046,      1, 123924, 117632, 119425, 113167, 120572,
                     113381,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [125488, 113398, 113165,      1, 113165,   9086, 116472,      1, 119791,
                     113604, 114109, 115046,      1, 123924, 117632, 119425, 113167, 120572,
                     113381,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [113306, 116408, 135535,      1,      1,      1, 137810, 176957, 129403,
                     123127, 177487,      1, 113249, 113236,      1, 113558,      1, 113199,
                     113472,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0],
                    [113306, 116408, 135535,      1,      1,      1, 137810, 176957, 129403,
                     123127, 177487,      1, 113249, 113236,      1, 113558,      1, 113199,
                     113472,      6,      0,      0,      0,      0,      0,      0,      0,
                          0,      0,      0,      0,      0,      0]], device='cuda:0')
            new_intpus['retrieve_ids']= tensor([[ 2,  3,  4,  6,  8, 10, 11, 12, 13, 16, 18, 19, 22, 24, 25, 26, 28, 29,
                     30, 31, 32, 34, 35, 36, 38, 40, 41, 42, 43, 46, 47, 49, 51],
                    [ 2,  3,  4,  6,  8, 10, 11, 12, 13, 16, 18, 19, 22, 24, 25, 26, 28, 29,
                     30, 31, 32, 34, 35, 36, 38, 40, 41, 42, 43, 46, 47, 49, 51],
                    [ 2,  3,  4,  6,  8, 10, 11, 12, 13, 16, 18, 19, 22, 24, 25, 26, 28, 29,
                     30, 31, 32, 34, 35, 36, 38, 40, 41, 42, 43, 46, 47, 49, 51],
                    [ 2,  3,  4,  6,  8, 10, 12, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27,
                     29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 44,  0,  0],
                    [ 2,  3,  4,  6,  8, 10, 12, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27,
                     29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 44,  0,  0],
                    [ 2,  3,  4,  6,  8, 10, 12, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27,
                     29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 44,  0,  0],
                    [ 2,  3,  4,  6,  8, 10, 12, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27,
                     29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 44,  0,  0],
                    [ 1,  2,  3,  4,  5,  6,  7, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26,
                     27, 30, 32, 33, 34, 38, 39, 40, 42,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  3,  4,  5,  6,  7, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26,
                     27, 30, 32, 33, 34, 38, 39, 40, 42,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  3,  4,  5,  6,  7, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26,
                     27, 30, 32, 33, 34, 38, 39, 40, 42,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  4,  5,  7,  8, 10, 12, 15, 16, 19, 20, 21, 24, 25, 26, 27, 29,
                     32, 34, 35, 37, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  4,  5,  7,  8, 10, 12, 15, 16, 19, 20, 21, 24, 25, 26, 27, 29,
                     32, 34, 35, 37, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  4,  7,  8, 10, 11, 14, 18, 20, 22, 23, 25, 26, 27, 29, 30, 31,
                     32, 34, 36, 39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  4,  7,  8, 10, 11, 14, 18, 20, 22, 23, 25, 26, 27, 29, 30, 31,
                     32, 34, 36, 39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  4,  7,  8, 10, 11, 14, 18, 20, 22, 23, 25, 26, 27, 29, 30, 31,
                     32, 34, 36, 39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  4,  7,  8, 10, 11, 14, 18, 20, 22, 23, 25, 26, 27, 29, 30, 31,
                     32, 34, 36, 39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  5,  7,  9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 27, 30, 32, 33,
                     34, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  5,  7,  9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 27, 30, 32, 33,
                     34, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  5,  7,  9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 27, 30, 32, 33,
                     34, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  3,  5,  7,  9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 27, 30, 32, 33,
                     34, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  7,  8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 28, 32,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  7,  8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 28, 32,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  3,  4,  8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 24, 25,
                     26, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  3,  4,  8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 24, 25,
                     26, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  4,  6,  7,  8,  9, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 1,  2,  4,  6,  7,  8,  9, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24,
                     25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                   device='cuda:0')
            new_inputs['upos_ids']= tensor([[ 7,  8,  4, 18, 18,  6,  4, 18, 18,  9,  2, 18, 18,  2,  6,  4,  7,  7,
                     14, 13,  8,  3,  4, 18,  8,  4,  9,  2,  4,  2, 18,  7,  7],
                    [ 7,  8,  4, 18, 18,  6,  4, 18, 18,  9,  2, 18, 18,  2,  6,  4,  7,  7,
                     14, 13,  8,  3,  4, 18,  8,  4,  9,  2,  4,  2, 18,  7,  7],
                    [ 7,  8,  4, 18, 18,  6,  4, 18, 18,  9,  2, 18, 18,  2,  6,  4,  7,  7,
                     14, 13,  8,  3,  4, 18,  8,  4,  9,  2,  4,  2, 18,  7,  7],
                    [ 7, 18,  8,  4, 18, 18, 18,  9,  2, 18,  2,  4,  4,  9,  2,  4,  4,  9,
                      2,  4,  4,  6,  4, 18,  7,  4,  9, 18,  9,  7,  7,  0,  0],
                    [ 7, 18,  8,  4, 18, 18, 18,  9,  2, 18,  2,  4,  4,  9,  2,  4,  4,  9,
                      2,  4,  4,  6,  4, 18,  7,  4,  9, 18,  9,  7,  7,  0,  0],
                    [ 7, 18,  8,  4, 18, 18, 18,  9,  2, 18,  2,  4,  4,  9,  2,  4,  4,  9,
                      2,  4,  4,  6,  4, 18,  7,  4,  9, 18,  9,  7,  7,  0,  0],
                    [ 7, 18,  8,  4, 18, 18, 18,  9,  2, 18,  2,  4,  4,  9,  2,  4,  4,  9,
                      2,  4,  4,  6,  4, 18,  7,  4,  9, 18,  9,  7,  7,  0,  0],
                    [ 8,  8, 14,  8,  4,  4,  4,  2, 18, 18,  9,  2,  4,  2,  4, 18, 18, 18,
                      4, 18, 18, 18, 18,  2,  4,  4,  7,  0,  0,  0,  0,  0,  0],
                    [ 8,  8, 14,  8,  4,  4,  4,  2, 18, 18,  9,  2,  4,  2,  4, 18, 18, 18,
                      4, 18, 18, 18, 18,  2,  4,  4,  7,  0,  0,  0,  0,  0,  0],
                    [ 8,  8, 14,  8,  4,  4,  4,  2, 18, 18,  9,  2,  4,  2,  4, 18, 18, 18,
                      4, 18, 18, 18, 18,  2,  4,  4,  7,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  4,  9,  9, 18, 18, 14, 18, 13, 12,  9, 14, 14,  8,  4,  9,
                      6, 14, 18,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  4,  9,  9, 18, 18, 14, 18, 13, 12,  9, 14, 14,  8,  4,  9,
                      6, 14, 18,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  8, 18, 14,  4,  4, 18, 18, 18,  2,  4,  4,  4,  2, 18,  2, 14,  2,
                      4,  2, 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  8, 18, 14,  4,  4, 18, 18, 18,  2,  4,  4,  4,  2, 18,  2, 14,  2,
                      4,  2, 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  8, 18, 14,  4,  4, 18, 18, 18,  2,  4,  4,  4,  2, 18,  2, 14,  2,
                      4,  2, 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  8, 18, 14,  4,  4, 18, 18, 18,  2,  4,  4,  4,  2, 18,  2, 14,  2,
                      4,  2, 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18,  2,  4,  2,  4,  4,  9, 18,  4,  4, 18,  9, 18, 18,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18,  2,  4,  2,  4,  4,  9, 18,  4,  4, 18,  9, 18, 18,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18,  2,  4,  2,  4,  4,  9, 18,  4,  4, 18,  9, 18, 18,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18,  2,  4,  2,  4,  4,  9, 18,  4,  4, 18,  9, 18, 18,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4,  2,  4,  9,  9,  2,  4,  2,  4,  9,  2,  4,  2,  9, 18,  7,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4,  2,  4,  9,  9,  2,  4,  2,  4,  9,  2,  4,  2,  9, 18,  7,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18, 18,  2,  4,  2, 18, 14,  4,  8, 14,  4,  9, 18,  4,  4,
                     18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  4, 18, 18,  2,  4,  2, 18, 14,  4,  8, 14,  4,  9, 18,  4,  4,
                     18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  9,  2,  6,  4,  2,  4,  2,  4, 18,  2,  4,  8,  4,  2,  4,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  9,  2,  6,  4,  2,  4,  2,  4, 18,  2,  4,  8,  4,  2,  4,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  9,  2,  6,  4,  2,  4,  2,  4, 18,  2,  4,  8,  4,  2,  4,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4,  2,  9,  2,  6,  4,  2,  4,  2,  4, 18,  2,  4,  8,  4,  2,  4,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4, 18,  9, 14, 18, 18,  7,  4,  4,  7,  2,  4,  4,  2,  4,  9,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 8,  4, 18,  9, 14, 18, 18,  7,  4,  4,  7,  2,  4,  4,  2,  4,  9,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                   device='cuda:0')
            new_inputs['xpos_ids']= tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
            new_inputs['head_ids']= tensor([[ 2,  0,  2,  3,  4,  2,  6,  9,  6,  2, 12, 10, 12, 15, 10, 15,  2, 21,
                     21, 21,  2, 23, 21, 25, 21, 25, 26, 29, 26, 31, 25, 25,  2],
                    [ 2,  0,  2,  3,  4,  2,  6,  9,  6,  2, 12, 10, 12, 15, 10, 15,  2, 21,
                     21, 21,  2, 23, 21, 25, 21, 25, 26, 29, 26, 31, 25, 25,  2],
                    [ 2,  0,  2,  3,  4,  2,  6,  9,  6,  2, 12, 10, 12, 15, 10, 15,  2, 21,
                     21, 21,  2, 23, 21, 25, 21, 25, 26, 29, 26, 31, 25, 25,  2],
                    [ 3,  3,  0,  3,  6,  3,  6,  7, 10,  6, 12,  3, 12, 13, 16, 12, 16, 17,
                     20, 18, 20, 20, 22, 23, 26, 20, 26, 26, 28,  3,  3,  0,  0],
                    [ 3,  3,  0,  3,  6,  3,  6,  7, 10,  6, 12,  3, 12, 13, 16, 12, 16, 17,
                     20, 18, 20, 20, 22, 23, 26, 20, 26, 26, 28,  3,  3,  0,  0],
                    [ 3,  3,  0,  3,  6,  3,  6,  7, 10,  6, 12,  3, 12, 13, 16, 12, 16, 17,
                     20, 18, 20, 20, 22, 23, 26, 20, 26, 26, 28,  3,  3,  0,  0],
                    [ 3,  3,  0,  3,  6,  3,  6,  7, 10,  6, 12,  3, 12, 13, 16, 12, 16, 17,
                     20, 18, 20, 20, 22, 23, 26, 20, 26, 26, 28,  3,  3,  0,  0],
                    [ 0,  1,  4,  2,  4,  5,  6,  9,  5,  9, 10, 13, 10, 15,  4, 17,  9, 17,
                     18, 21, 19, 21, 19, 25, 23, 25,  1,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  4,  5,  6,  9,  5,  9, 10, 13, 10, 15,  4, 17,  9, 17,
                     18, 21, 19, 21, 19, 25, 23, 25,  1,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  4,  5,  6,  9,  5,  9, 10, 13, 10, 15,  4, 17,  9, 17,
                     18, 21, 19, 21, 19, 25, 23, 25,  1,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  4,  4,  2,  7, 13, 13, 13, 13,  1, 16, 14, 13, 16, 17,
                     18, 21, 16, 21,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  4,  4,  2,  7, 13, 13, 13, 13,  1, 16, 14, 13, 16, 17,
                     18, 21, 16, 21,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  0,  2,  8,  8,  5,  5,  2,  8, 11,  8, 11, 12, 15, 11, 19, 19, 19,
                     11, 21, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  0,  2,  8,  8,  5,  5,  2,  8, 11,  8, 11, 12, 15, 11, 19, 19, 19,
                     11, 21, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  0,  2,  8,  8,  5,  5,  2,  8, 11,  8, 11, 12, 15, 11, 19, 19, 19,
                     11, 21, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 2,  0,  2,  8,  8,  5,  5,  2,  8, 11,  8, 11, 12, 15, 11, 19, 19, 19,
                     11, 21, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  3,  6,  3,  8,  1,  8,  8, 12,  1, 12, 13, 13, 17, 15, 19,
                     12,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  3,  6,  3,  8,  1,  8,  8, 12,  1, 12, 13, 13, 17, 15, 19,
                     12,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  3,  6,  3,  8,  1,  8,  8, 12,  1, 12, 13, 13, 17, 15, 19,
                     12,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  3,  6,  3,  8,  1,  8,  8, 12,  1, 12, 13, 13, 17, 15, 19,
                     12,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  1,  5,  1,  5,  5,  9,  1, 11,  9, 11, 14, 11, 16, 14, 16,  1,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  1,  5,  1,  5,  5,  9,  1, 11,  9, 11, 14, 11, 16, 14, 16,  1,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  5,  2,  7,  1,  9,  7, 12, 12,  1, 16, 16, 14, 12, 16, 17,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  5,  2,  7,  1,  9,  7, 12, 12,  1, 16, 16, 14, 12, 16, 17,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  6,  4,  6,  9,  1, 11,  9, 11, 14,  1, 14, 15, 18, 16,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  6,  4,  6,  9,  1, 11,  9, 11, 14,  1, 14, 15, 18, 16,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  6,  4,  6,  9,  1, 11,  9, 11, 14,  1, 14, 15, 18, 16,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  4,  2,  6,  4,  6,  9,  1, 11,  9, 11, 14,  1, 14, 15, 18, 16,
                     18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  2,  7,  7,  1,  7,  1,  9, 13, 13,  9, 13, 16, 13, 16, 19,
                      9,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 0,  1,  2,  2,  7,  7,  1,  7,  1,  9, 13, 13,  9, 13, 16, 13, 16, 19,
                      9,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                   device='cuda:0')
            new_inputs['deprel_ids']= tensor([[ 7,  9,  8,  5, 24,  4,  5, 20,  5, 18,  2, 13, 24,  2,  4,  5,  7,  7,
                     17, 13, 26,  3,  8, 20, 12,  8, 15,  2,  5,  2,  4,  7,  7],
                    [ 7,  9,  8,  5, 24,  4,  5, 20,  5, 18,  2, 13, 24,  2,  4,  5,  7,  7,
                     17, 13, 26,  3,  8, 20, 12,  8, 15,  2,  5,  2,  4,  7,  7],
                    [ 7,  9,  8,  5, 24,  4,  5, 20,  5, 18,  2, 13, 24,  2,  4,  5,  7,  7,
                     17, 13, 26,  3,  8, 20, 12,  8, 15,  2,  5,  2,  4,  7,  7],
                    [ 7, 16,  9,  8,  2,  4,  5, 15,  2,  5,  2,  4,  5, 15,  2,  4,  5, 15,
                      2,  4,  5,  6,  5, 20,  7,  5, 15,  5, 15,  7,  7,  0,  0],
                    [ 7, 16,  9,  8,  2,  4,  5, 15,  2,  5,  2,  4,  5, 15,  2,  4,  5, 15,
                      2,  4,  5,  6,  5, 20,  7,  5, 15,  5, 15,  7,  7,  0,  0],
                    [ 7, 16,  9,  8,  2,  4,  5, 15,  2,  5,  2,  4,  5, 15,  2,  4,  5, 15,
                      2,  4,  5,  6,  5, 20,  7,  5, 15,  5, 15,  7,  7,  0,  0],
                    [ 7, 16,  9,  8,  2,  4,  5, 15,  2,  5,  2,  4,  5, 15,  2,  4,  5, 15,
                      2,  4,  5,  6,  5, 20,  7,  5, 15,  5, 15,  7,  7,  0,  0],
                    [ 9, 18, 17, 27,  8,  5,  5,  2,  5,  5, 15,  2,  5,  2,  4, 20, 12,  5,
                      5,  2,  5, 24, 20,  2,  4,  5,  7,  0,  0,  0,  0,  0,  0],
                    [ 9, 18, 17, 27,  8,  5,  5,  2,  5,  5, 15,  2,  5,  2,  4, 20, 12,  5,
                      5,  2,  5, 24, 20,  2,  4,  5,  7,  0,  0,  0,  0,  0,  0],
                    [ 9, 18, 17, 27,  8,  5,  5,  2,  5,  5, 15,  2,  5,  2,  4, 20, 12,  5,
                      5,  2,  5, 24, 20,  2,  4,  5,  7,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2,  5, 15, 15,  5, 24, 17,  8, 13, 25, 22, 17, 28, 26,  8, 15,
                      6, 20, 12, 10,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2,  5, 15, 15,  5, 24, 17,  8, 13, 25, 22, 17, 28, 26,  8, 15,
                      6, 20, 12, 10,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  9,  8, 17,  8,  5,  5, 10, 10,  2,  4,  5,  5,  2, 13,  2, 17,  2,
                      5,  2,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  9,  8, 17,  8,  5,  5, 10, 10,  2,  4,  5,  5,  2, 13,  2, 17,  2,
                      5,  2,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  9,  8, 17,  8,  5,  5, 10, 10,  2,  4,  5,  5,  2, 13,  2, 17,  2,
                      5,  2,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 7,  9,  8, 17,  8,  5,  5, 10, 10,  2,  4,  5,  5,  2, 13,  2, 17,  2,
                      5,  2,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  2,  5,  2,  4,  5, 15,  2,  4,  5,  5, 15,  2,  5,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  2,  5,  2,  4,  5, 15,  2,  4,  5,  5, 15,  2,  5,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  2,  5,  2,  4,  5, 15,  2,  4,  5,  5, 15,  2,  5,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  2,  5,  2,  4,  5, 15,  2,  4,  5,  5, 15,  2,  5,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8, 10,  2,  4, 15, 15,  2,  4,  2,  5, 15,  2,  5,  2, 15, 20,  7,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8, 10,  2,  4, 15, 15,  2,  4,  2,  5, 15,  2,  5,  2, 15, 20,  7,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  5,  2,  4,  2,  5, 17,  8, 22, 17,  8, 15, 10, 10,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5,  5,  5,  2,  4,  2,  5, 17,  8, 22, 17,  8, 15, 10, 10,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2, 15,  2,  6,  5,  2,  4,  2,  5,  5,  2,  4, 14, 10,  2,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2, 15,  2,  6,  5,  2,  4,  2,  5,  5,  2,  4, 14, 10,  2,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2, 15,  2,  6,  5,  2,  4,  2,  5,  5,  2,  4, 14, 10,  2,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  2, 15,  2,  6,  5,  2,  4,  2,  5,  5,  2,  4, 14, 10,  2,  5,
                      5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5, 15, 17,  5, 10,  7, 10,  5,  7,  2,  5,  5,  2,  5, 15,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 9,  8,  5, 15, 17,  5, 10,  7, 10,  5,  7,  2,  5,  5,  2,  5, 15,  2,
                      4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                   device='cuda:0')
            new_inputs['ner_ids']= tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
            new_inputs['trigger']= tensor([ 1, 22, 24,  2,  3, 11, 23,  4, 12, 25,  0, 16,  1,  8, 10, 17,  3,  7,
                     8, 11,  2, 10,  0,  6,  0,  8, 13, 15,  0, 18])
            new_inputs['eid']= tensor([1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 5, 5, 2, 2, 2, 2, 9, 9, 9, 9, 7, 7, 8, 8,
                    4, 4, 4, 4, 0, 0])
            new_inputs['pad_masks']= tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
            """

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

            """            
            new_inputs['biw2v_ids'].shape= torch.Size([30, 33])
            new_intpus['retrieve_ids'].shape= torch.Size([30, 33])
            new_inputs['upos_ids'].shape= torch.Size([30, 33])
            new_inputs['xpos_ids'].shape= torch.Size([30, 33])
            new_inputs['head_ids'].shape= torch.Size([30, 33])
            new_inputs['deprel_ids'].shape= torch.Size([30, 33])
            new_inputs['ner_ids'].shape= torch.Size([30, 33])
            new_inputs['pad_masks'].shape= torch.Size([30, 33])
            """

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
