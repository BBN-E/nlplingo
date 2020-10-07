import logging, os
import argparse
import torch, re


def parse_str_list(input_):
    if input_ == None:
        return []
    return list(map(str, input_.split(',')))


# class MyLogger:
#     def __init__(self, output_path):
#         self.output_path = output_path
#         with open(output_path, 'w') as f:
#             f.write('-' * 50 + ' Begin logging ' + '-' * 50 + '\n')
#
#     def info(self, string):
#         with open(self.output_path, 'a') as f:
#             f.write(string + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--biw2v_map_dir', type=str)		# ==>
parser.add_argument('--datapoint_dir', type=str)		# ==>
parser.add_argument('--stanford_resource_dir', type=str)	# ==>
parser.add_argument('--log_dir', type=str)			# ==>
parser.add_argument('--log_name', type=str, default='training_log')
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--upos_dim', default=30, type=int)
parser.add_argument('--xpos_dim', default=30, type=int)
parser.add_argument('--deprel_dim', default=30, type=int)
parser.add_argument('--dist_dim', default=30, type=int)
parser.add_argument('--ner_dim', default=30, type=int)
parser.add_argument('--num_last_layer_xlmr', default=1, type=int)
parser.add_argument('--hidden_dim', default=200, type=int)
parser.add_argument('--cross_valid', default='', type=str,
                    help='for cross-validation, put name of fold, e.g. fold1 or fold2')
parser.add_argument('--data', default='abstract', type=str,
                    help='for cross-validation, put name of fold, e.g. fold1 or fold2'
                         'for simulated data, put name of setting, e.g., en-en or en-ar')
parser.add_argument('--max_grad_norm', default=5.0, type=float)
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.', type=str)
parser.add_argument('--lr', default=0.00002, type=float)
parser.add_argument('--num_epoch', default=60, type=int)
parser.add_argument('--trainer', default='trigger', type=str)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--context_layer', default='lstm', choices=['lstm', 'gcn'], type=str)
parser.add_argument('--gcn_dropout', default=0.5, type=float)
parser.add_argument('--prune_tree', default=0, type=int)
parser.add_argument('--model', default='pipeline-01', choices=['pipeline-01', 'joint-01'], type=str)
# ******** bivec params ************************************
parser.add_argument('--use_biw2v', default=0, type=int)
parser.add_argument('--finetune_biw2v', default=0, type=int)
# ******** self attention params ********************************
parser.add_argument('--self_att_layers', default=6, type=int)
parser.add_argument('--self_att_heads', default=1, type=int)
parser.add_argument('--self_att_d_qkv', default=200, type=int)
parser.add_argument('--self_att_dropout', default=0.1, type=float)
parser.add_argument('--lstm_by_satt', default=0, type=int, help='Replace LSTM by Self-attention')
parser.add_argument('--lstm_add_satt', default=0, type=int, help='Replace LSTM by LSTM+Self-attention')
parser.add_argument('--lambda_mix', default=0.8, type=float)
# ******** LSTM ************************************************
parser.add_argument('--lstm_layers_trigger', default=4, type=int)
parser.add_argument('--lstm_layers_entity', default=1, type=int)
parser.add_argument('--lstm_layers_event', default=1, type=int)
# ******** Others **********************************************
parser.add_argument('--use_dep_edge', default=0, type=int)
parser.add_argument('--use_cased_entity', default=1, type=int)
parser.add_argument('--use_elmo', default=0, type=int)
parser.add_argument('--do_exp', default='default', type=str)
# ************** Pipeline setting ******************************
parser.add_argument('--train_file', default='app/train_data.bp.json', type=str)
parser.add_argument('--dev_file', default='datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json',
                    type=str)
parser.add_argument('--test_file', default=None, type=str)
parser.add_argument('--output_format', default='json', type=str, choices=['json', 'txt'])
parser.add_argument('--output_file', default=None, type=str)
parser.add_argument('--input_lang', default=None, type=str, choices=['english', 'arabic'])
parser.add_argument('--data_map', default=None, type=str)
parser.add_argument('--inhouse_eval', default=0,
                    type=int)  # set to 1 if using official scorer to self-evaluate performance
parser.add_argument('--ensemble_mode', default=0, type=int)
parser.add_argument('--ensemble_seeds', default=['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025'],
                    type=parse_str_list)
parser.add_argument('--finetune_xlmr', default=1, type=int)
parser.add_argument('--dropout_xlmr', default=0.1, type=float)
parser.add_argument('--train_ED', default=1, type=int)
parser.add_argument('--train_argument', default=1, type=int)
parser.add_argument('--get_perf_of_separate_models', default=0, type=int)
parser.add_argument('--edge_lambda', default=0.1, type=float)
parser.add_argument('--use_dep2sent', default=0, type=int)
parser.add_argument('--xlmr_version', default='xlmr.base', type=str, choices=['xlmr.large', 'xlmr.base'])
parser.add_argument('--xlmr_model_dir', type=str)		# ==>
parser.add_argument('--position_embed_for_satt', default=1, type=int)
parser.add_argument('--ED_eval_epoch', default=0, type=int)
parser.add_argument('--argument_eval_epoch', default=0, type=int)
parser.add_argument('--output_offsets', default=1, type=int)
parser.add_argument('--train_strategy', default='retrain.add-all', type=str,
                    help='strategy to deal with additional hidden training data',
                    choices=['retrain.add-all', 'retrain.add-bad', 'cont_train.all',
                             'cont_train.bad'])
parser.add_argument('--delete_nonbest_ckpts', default=1, type=int)
parser.add_argument('--docker_run', default=1, type=int)
parser.add_argument('--hidden_eval', default=0,
                    type=int)  # turn on when you want to output probabilities of predictions, used for evaluation of hidden_eval.py
parser.add_argument('--bad_threshold', default=0.4, type=float)  # threshold to select hidden training data
parser.add_argument('--readers_mode', default=1, type=int)
parser.add_argument('--use_ner', default=0, type=int)
parser.add_argument('--initialize_with_pretrained', default=0,
                    type=int)  # initialize models with pretrained weights
parser.add_argument('--train_on_arb', default=0, type=int)
parser.add_argument('--remove_incomplete',
                    default=0,
                    type=int)  # set to 1 if you want to remove arb sentences with missing events when generating arb-wa-corpus
parser.add_argument('--co_train_lambda', default=0, type=float)  # coeff for loss on generated arb training instances.
parser.add_argument('--finetune_on_arb', default=0, type=int)  # only update xlmr embeddings, fix classification layers
parser.add_argument('--num_first_xlmr_layers', default=4, type=int)  # embedding layer + 24 xlmr layers
parser.add_argument('--save_last_epoch', default=1, type=int)
parser.add_argument('--grad_clip_xlmr', default=0, type=int)

# for NLPLingo
parser.add_argument('--mode')   # ==>
parser.add_argument('--params')	# ==>

def make_opt():
    WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    opt = parser.parse_args()
    opt = vars(opt)

    """
    {'log_name': 'train.log.ed', 'device': None, 'batch_size': 16, 'upos_dim': 30, 'xpos_dim': 30, 'deprel_dim': 30,
     'dist_dim': 30, 'ner_dim': 30, 'num_last_layer_xlmr': 1, 'hidden_dim': 200, 'cross_valid': '', 'data': 'abstract',
     'max_grad_norm': 5.0, 'optim': 'adam', 'lr': 5e-06, 'num_epoch': 5, 'trainer': 'trigger', 'seed': 2020,
     'context_layer': 'lstm', 'gcn_dropout': 0.5, 'prune_tree': 0, 'model': 'pipeline-01', 'use_biw2v': 0,
     'finetune_biw2v': 0, 'self_att_layers': 6, 'self_att_heads': 1, 'self_att_d_qkv': 200, 'self_att_dropout': 0.1,
     'lstm_by_satt': 0, 'lstm_add_satt': 0, 'lambda_mix': 0.8, 'lstm_layers_trigger': 4, 'lstm_layers_entity': 1,
     'lstm_layers_event': 1, 'use_dep_edge': 0, 'use_cased_entity': 1, 'use_elmo': 0, 'do_exp': 'default',
     'train_file': None, 'dev_file': 'datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json',
     'test_file': None, 'output_format': 'json', 'output_file': None, 'input_lang': 'english', 'data_map': None,
     'inhouse_eval': 0, 'ensemble_mode': 0,
     'ensemble_seeds': ['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025'], 'finetune_xlmr': 1,
     'dropout_xlmr': 0.5, 'train_ED': 1, 'train_argument': 0, 'get_perf_of_separate_models': 0, 'edge_lambda': 0.1,
     'use_dep2sent': 0, 'xlmr_version': 'xlmr.base', 'position_embed_for_satt': 1, 'ED_eval_epoch': 0,
     'argument_eval_epoch': 0, 'output_offsets': 1, 'train_strategy': 'retrain.add-all', 'delete_nonbest_ckpts': 1,
     'docker_run': 0, 'hidden_eval': 0, 'bad_threshold': 0.4, 'readers_mode': 1, 'use_ner': 0,
     'initialize_with_pretrained': 0, 'train_on_arb': 0, 'remove_incomplete': 0, 'co_train_lambda': 0,
     'finetune_on_arb': 0, 'num_first_xlmr_layers': 7, 'save_last_epoch': 1}
    """

    opt['train_is_dir'] = True
    opt['test_is_dir'] = False

    if opt['test_file'] and os.path.isdir(opt['test_file']):
        opt['test_is_dir'] = True
        if not os.path.exists(opt['output_file']):
            os.mkdir(opt['output_file'])
    elif opt['test_file'] and opt['output_file'] is None:
        opt['output_file'] = opt['test_file'] + '.system-output'

    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt['device'] is None else torch.device(opt['device'])

    assert 0 <= opt['edge_lambda'] <= 1.0

    opt['finetuned_xlmr_layers'] = []
    if opt['num_first_xlmr_layers'] >= 1:
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.embed_tokens')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.embed_positions')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.embed_tokens')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.embed_positions')
        opt['finetuned_xlmr_layers'].append('self_att.attention_layers')
        opt['finetuned_xlmr_layers'].append('gcn_layer')
        opt['finetuned_xlmr_layers'].append('biw2v_embedding')
        for k in range(opt['num_first_xlmr_layers'] - 1):
            opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.layers.{}.'.format(k))
            opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.layers.{}.'.format(k))

    # log_dir = os.path.join(WORKING_DIR, 'logs')	# <==
    # log_dir = opt['log_dir']				# ==>
    # if opt['docker_run']:
    #     opt['ckpt_dir'] = os.path.join('/code/checkpoints')
    # else:
    #     opt['ckpt_dir'] = os.path.join('checkpoints')

    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # if not os.path.exists(opt['ckpt_dir']):
    #     os.mkdir(opt['ckpt_dir'])
    # logger = MyLogger(output_path=os.path.join(log_dir, opt['log_name']))
    return opt

opt = make_opt()
