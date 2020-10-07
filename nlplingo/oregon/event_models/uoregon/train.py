import random, time, torch
import numpy as np
from nlplingo.oregon.event_models.uoregon.define_opt import opt
from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.readers import read_abstract_train_data, \
    read_abstract_train_data_from_files

if opt['train_is_dir']:
    hidden_train_files = get_files_in_dir_recur(opt['train_file'])  # additional hidden training data
else:
    hidden_train_files = []

assert opt['hidden_eval'] == 0  # only set this to 1 when using hidden_eval.py
assert opt['train_on_arb'] == 0  # only set this to 1 when using pretrain_on_arb.py
assert opt['input_lang'] == 'english'

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
 'initialize_with_pretrained': 0, 'train_on_arb': 0, 'remove_incomplete': 0, 'co_train_lambda': 0, 'finetune_on_arb': 0,
 'num_first_xlmr_layers': 7, 'save_last_epoch': 1}
"""

if len(hidden_train_files) == 0:
    logger.info('No hidden training files')
    opt['train_strategy'] = 'retrain.add-all'

    #opt['observed_train'] = 'datasets/8d/update2/abstract-8d-inclusive.train.update2.bp.json'		# <==
    opt['observed_train'] = 'datasets/8d/update2/abstract-8d-inclusive.small-train.update2.bp.json'	# ==>
    opt['dev_file'] = 'datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json'

    """ YS
    The following generates the JSON representation of each sentence, in the directory: 
    ./python/clever/event_models/uoregon/models/pipeline/_01/data/abstract-8d-inclusive
    
    Each file is a list. Each element in list is the JSON representation of a Sentence.
    E.g. for trigger, following is an element:
    {
    "id": "doc-97_25_0", 
    "entry_id": "doc-97_25_0", 
    "triggers": [[6], [8]], 
    "event-types": ["harmful|verbal", "neutral|verbal"], 
    "text": "Mr. Rabbani and other opposition leaders disputed the claims by government officials.", 
    "ori_text": "Mr. Rabbani and other opposition leaders disputed the claims by government officials.", 
    "ori_entry": {
        "annotation-sets": {
            "abstract-events": {
                "events": {
                    "event1": {
                        "agents": ["ss-4", "ss-5"], 
                        "anchors": "ss-3", 
                        "eventid": "event1", 
                        "helpful-harmful": "harmful", 
                        "material-verbal": "verbal", 
                        "patients": ["ss-1"], 
                        "anchor_offsets": {}, 
                        "agent_offsets": {}, 
                        "patient_offsets": {}
                    }, 
                    "event2": {
                        "agents": ["ss-2"], 
                        "anchors": "ss-1", 
                        "eventid": "event2", 
                        "helpful-harmful": "neutral", 
                        "material-verbal": "verbal", 
                        "patients": [], 
                        "anchor_offsets": {}, 
                        "agent_offsets": {}, 
                        "patient_offsets": {}
                    }
                }, 
                "span-sets": {
                    "ss-1": {
                        "spans": [{"string": "claims"}], 
                        "ssid": "ss-1"
                    }, 
                    "ss-2": {
                        "spans": [{"string": "government officials"}], 
                        "ssid": "ss-2"
                    }, 
                    "ss-3": {
                        "spans": [{"string": "disputed"}], 
                        "ssid": "ss-3"
                    }, 
                    "ss-4": {
                        "spans": [{"string": "Mr. Rabbani"}], 
                        "ssid": "ss-4"
                    }, 
                    "ss-5": {
                        "spans": [{"string": "other opposition leaders"}], 
                        "ssid": "ss-5"
                    }
                }
            }
        }, 
        "doc-id": "doc-97", 
        "entry-id": "doc-97_25_0", 
        "segment-text": "Mr. Rabbani and other opposition leaders disputed the claims by government officials.", 
        "segment-type": "sentence", 
        "sent-id": "25"
    }, 
    "norm2ori_offsetmap": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38, "39": 39, "40": 40, "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47, "48": 48, "49": 49, "50": 50, "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56, "57": 57, "58": 58, "59": 59, "60": 60, "61": 61, "62": 62, "63": 63, "64": 64, "65": 65, "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "72": 72, "73": 73, "74": 74, "75": 75, "76": 76, "77": 77, "78": 78, "79": 79, "80": 80, "81": 81, "82": 82, "83": 83, "84": 84}, 
    "lang": "english", 
    "word": ["Mr.", "Rabbani", "and", "other", "opposition", "leaders", "disputed", "the", "claims", "by", "government", "officials", "."], 
    "lemma": ["Mr.", "Rabbani", "and", "other", "opposition", "leader", "dispute", "the", "claim", "by", "government", "official", "."], 
    "upos": ["PROPN", "PROPN", "CCONJ", "ADJ", "NOUN", "NOUN", "VERB", "DET", "NOUN", "ADP", "NOUN", "NOUN", "PUNCT"], 
    "xpos": ["NNP", "NNP", "CC", "JJ", "NN", "NNS", "VBD", "DT", "NNS", "IN", "NN", "NNS", "."], 
    "morph": ["Number=Sing", "Number=Sing", "_", "Degree=Pos", "Number=Sing", "Number=Plur", "Mood=Ind|Tense=Past|VerbForm=Fin", "Definite=Def|PronType=Art", "Number=Plur", "_", "Number=Sing", "Number=Plur", "_"], 
    "head": [7, 1, 6, 6, 6, 1, 0, 9, 7, 12, 12, 9, 7], 
    "dep_rel": ["nsubj", "flat", "cc", "amod", "compound", "conj", "root", "det", "obj", "case", "compound", "nmod", "punct"], 
    "ner": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], 
    "span": [[0, 2], [4, 10], [12, 14], [16, 20], [22, 31], [33, 39], [41, 48], [50, 52], [54, 59], [61, 62], [64, 73], [75, 83], [84, 84]]
    }
    """
    data_map = read_abstract_train_data(opt['observed_train'], opt['dev_file'])

    opt['data_map'] = data_map

    from python.clever.event_models.uoregon.models.pipeline._01.trainers import *

    # ************* ED model *****************
    if opt['train_ED']:
        torch.autograd.set_detect_anomaly(True)
        # YS: following are for reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(opt['seed'])
        np.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed(opt['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ED_trainer = EDTrainer(opt)
        if opt['get_perf_of_separate_models']:
            ED_trainer.eval_with_saved_model()
        else:
            ED_trainer.train()
    # ************* argument model *****************
    if opt['train_argument']:
        torch.autograd.set_detect_anomaly(True)
        random.seed(opt['seed'])
        np.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed(opt['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        arg_trainer = ArgumentTrainer(opt)
        if opt['get_perf_of_separate_models']:
            arg_trainer.eval_with_saved_model()
        else:
            arg_trainer.train()
    if not opt['get_perf_of_separate_models']:
        print('Training done!')
else:
    logger.info('Hidden training files:\n{}\n'.format('\n'.join(hidden_train_files)))
    train_files = []

    opt['observed_train'] = 'datasets/8d/update2/abstract-8d-inclusive.train.update2.bp.json'

    train_files.append(opt['observed_train'])  # existing training data
    if opt['train_strategy'].startswith('retrain'):  # add-all or add-bad
        train_files += hidden_train_files

        data_map = read_abstract_train_data_from_files(train_files, opt['dev_file'], observed_included=True)
        opt['data_map'] = data_map

        from python.clever.event_models.uoregon.models.pipeline._01.trainers import *

        print('Start training with strategy {}'.format(opt['train_strategy']))
        logger.info('Start training with strategy {}'.format(opt['train_strategy']))
        # ************* ED model *****************
        if opt['train_ED']:
            torch.autograd.set_detect_anomaly(True)
            random.seed(opt['seed'])
            np.random.seed(opt['seed'])
            torch.manual_seed(opt['seed'])
            torch.cuda.manual_seed(opt['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            ED_trainer = EDTrainer(opt)
            if opt['get_perf_of_separate_models']:
                ED_trainer.eval_with_saved_model()
            else:
                ED_trainer.train()
        # ************* argument model *****************
        if opt['train_argument']:
            torch.autograd.set_detect_anomaly(True)
            random.seed(opt['seed'])
            np.random.seed(opt['seed'])
            torch.manual_seed(opt['seed'])
            torch.cuda.manual_seed(opt['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            arg_trainer = ArgumentTrainer(opt)
            if opt['get_perf_of_separate_models']:
                arg_trainer.eval_with_saved_model()
            else:
                arg_trainer.train()
        if not opt['get_perf_of_separate_models']:
            print('Training done!')
    elif opt['train_strategy'].startswith(
            'cont_train'):  # continuous training, start from saved model produced by hidden_eval.py
        print('Start training with strategy {}'.format(opt['train_strategy']))
        logger.info('Start training with strategy {}'.format(opt['train_strategy']))
        #################### continue training on hidden data #################
        data_map = read_abstract_train_data_from_files(hidden_train_files, opt['dev_file'], observed_included=False)
        opt['data_map'] = data_map

        from python.clever.event_models.uoregon.models.pipeline._01.trainers import *

        # ************* ED model *****************
        if opt['train_ED']:
            torch.autograd.set_detect_anomaly(True)
            random.seed(opt['seed'])
            np.random.seed(opt['seed'])
            torch.manual_seed(opt['seed'])
            torch.cuda.manual_seed(opt['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            ED_trainer = EDTrainer(opt)
            if opt['get_perf_of_separate_models']:
                ED_trainer.eval_with_saved_model()
            else:
                ED_trainer.train()
        # ************* argument model *****************
        if opt['train_argument']:
            torch.autograd.set_detect_anomaly(True)
            random.seed(opt['seed'])
            np.random.seed(opt['seed'])
            torch.manual_seed(opt['seed'])
            torch.cuda.manual_seed(opt['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            arg_trainer = ArgumentTrainer(opt)
            if opt['get_perf_of_separate_models']:
                arg_trainer.eval_with_saved_model()
            else:
                arg_trainer.train()
        if not opt['get_perf_of_separate_models']:
            print('Training done!')
