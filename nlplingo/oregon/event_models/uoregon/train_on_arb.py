import random, time, torch
import numpy as np
from nlplingo.oregon.event_models.uoregon.define_opt import opt
from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.readers import read_abstract_train_data
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.trainers import *

opt['train_on_arb'] = 1
opt['train_strategy'] = 'retrain.add-all'
opt['initialize_with_pretrained'] = 1
opt['finetune_on_arb'] = 1

opt['observed_train'] = 'datasets/8d/update2/arabic-wa-corpus.bp.json'
opt['dev_file'] = 'datasets/8d/update2/arabic-abstract-sample.bp.json'

assert opt['co_train_lambda'] == 0

assert opt['input_lang'] == 'arabic'

""" opt:
    ED_eval_epoch : 0
    argument_eval_epoch : 0
    bad_threshold : 0.4
    batch_size : 16
    biw2v_map_dir : resources/aligned_w2v
    biw2v_size : 354186
    biw2v_vecs : [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
       0.00000000e+00  0.00000000e+00]
     [ 8.26119033e-01  3.68800311e-01  8.69561242e-01 ...  2.70505650e-01
       2.05427664e-01  2.01526267e-01]
     [ 1.33400000e-03  1.47300000e-03 -1.27700000e-03 ... -4.37000000e-04
      -5.52000000e-04  1.02400000e-03]
     ...
     [-1.15833000e-01 -8.17270000e-02 -5.58370000e-02 ... -1.59482000e-01
      -3.43660000e-02  6.65400000e-03]
     [-3.82970000e-02 -5.19210000e-02 -7.23600000e-02 ... -1.40313000e-01
       1.73640000e-02  1.28790000e-02]
     [-1.11085000e-01 -4.86380000e-02 -8.37620000e-02 ... -1.55592000e-01
       6.28500000e-03  3.66210000e-02]]
    ckpt_dir : checkpoints
    co_train_lambda : 0
    context_layer : lstm
    cross_valid : 
    data : abstract
    data_map : None
    datapoint_dir : datapoints
    delete_nonbest_ckpts : 1
    deprel_dim : 30
    dev_file : datasets/8d/update2/arabic-abstract-sample.bp.json
    device : cuda
    dist_dim : 30
    do_exp : default
    docker_run : 0
    dropout_xlmr : 0.1
    edge_lambda : 0.1
    ensemble_mode : 0
    ensemble_seeds : ['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025']
    finetune_biw2v : 0
    finetune_on_arb : 1
    finetune_xlmr : 1
    finetuned_xlmr_layers : ['xlmr_embedding.model.decoder.sentence_encoder.embed_tokens', 
        'xlmr_embedding.model.decoder.sentence_encoder.embed_positions', 'self_att.attention_layers', 'gcn_layer', 
        'biw2v_embedding', 'xlmr_embedding.model.decoder.sentence_encoder.layers.0.', 
        'xlmr_embedding.model.decoder.sentence_encoder.layers.1.', 
        'xlmr_embedding.model.decoder.sentence_encoder.layers.2.', 
        'xlmr_embedding.model.decoder.sentence_encoder.layers.3.']
    gcn_dropout : 0.5
    get_perf_of_separate_models : 0
    grad_clip_xlmr : 0
    hidden_dim : 200
    hidden_eval : 0
    inhouse_eval : 0
    initialize_with_pretrained : 1
    input_lang : arabic
    lambda_mix : 0.8
    log_dir : logs
    log_name : train.log.arg.arabic-wa-corpus
    lr : 2e-05
    lstm_add_satt : 0
    lstm_by_satt : 0
    lstm_layers_entity : 1
    lstm_layers_event : 1
    lstm_layers_trigger : 4
    max_grad_norm : 5.0
    mode : None
    model : pipeline-01
    ner_dim : 30
    num_epoch : 10
    num_first_xlmr_layers : 5
    num_last_layer_xlmr : 1
    observed_train : datasets/8d/update2/arabic-wa-corpus.bp.json
    optim : adam
    output_file : None
    output_format : json
    output_offsets : 1
    params : None
    position_embed_for_satt : 1
    prune_tree : 0
    readers_mode : 1
    remove_incomplete : 0
    save_last_epoch : 1
    seed : 2020
    self_att_d_qkv : 200
    self_att_dropout : 0.1
    self_att_heads : 1
    self_att_layers : 6
    stanford_resource_dir : resources/stanford
    test_file : None
    test_is_dir : False
    train_ED : 0
    train_argument : 1
    train_file : app/train_data.bp.json
    train_is_dir : True
    train_on_arb : 1
    train_strategy : retrain.add-all
    trainer : trigger
    upos_dim : 30
    use_biw2v : 0
    use_cased_entity : 1
    use_dep2sent : 0
    use_dep_edge : 0
    use_elmo : 0
    use_ner : 0
    xlmr_model_dir : models/xlmr.base
    xlmr_version : xlmr.base
    xpos_dim : 30
"""

data_map = read_abstract_train_data(opt['observed_train'], opt['dev_file'])

opt['data_map'] = data_map

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
