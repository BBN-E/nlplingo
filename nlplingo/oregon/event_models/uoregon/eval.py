import random, time, torch
import numpy as np
from nlplingo.oregon.event_models.uoregon.define_opt import opt
from nlplingo.oregon.event_models.uoregon.tools.utils import *

if opt['docker_run'] == 1:
    opt['hidden_eval'] = 0

print(sorted(opt))
"""
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$pythonpath" "$python" "$uoregon"/python/clever/event_models/uoregon/eval.py 
    --input_lang arabic --output_offsets 1 --test_file ./datasets/8d/update2/ten_arabic_sentences.bp.v2.json 
    --output_file ./output/new-arabic.oregon.small-train.json --log_name eval.log.new-arabic.oregon.small-train 
    --ED_eval_epoch 20 --argument_eval_epoch 20 --docker_run 0 --xlmr_model_dir models/xlmr.base --log_dir logs 
    --stanford_resource_dir resources/stanford --datapoint_dir datapoints --biw2v_map_dir resources/aligned_w2v

    ED_eval_epoch : 20
    argument_eval_epoch : 20
    bad_threshold : 0.4
    batch_size : 16
    biw2v_map_dir : resources/aligned_w2v
    ckpt_dir : checkpoints
    co_train_lambda : 0
    context_layer : lstm
    cross_valid : 
    data : abstract
    data_map : None
    datapoint_dir : datapoints
    delete_nonbest_ckpts : 1
    deprel_dim : 30
    dev_file : datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json
    device : cuda
    dist_dim : 30
    do_exp : default
    docker_run : 0
    dropout_xlmr : 0.1
    edge_lambda : 0.1
    ensemble_mode : 0
    ensemble_seeds : ['seed-2021', 'seed-2022', 'seed-2023', 'seed-2024', 'seed-2025']
    finetune_biw2v : 0
    finetune_on_arb : 0
    finetune_xlmr : 1
    finetuned_xlmr_layers : ['xlmr_embedding.model.decoder.sentence_encoder.embed_tokens', 'xlmr_embedding.model.decoder.sentence_encoder.embed_positions', 'self_att.attention_layers', 'gcn_layer', 'biw2v_embedding', 'xlmr_embedding.model.decoder.sentence_encoder.layers.0.', 'xlmr_embedding.model.decoder.sentence_encoder.layers.1.', 'xlmr_embedding.model.decoder.sentence_encoder.layers.2.']
    gcn_dropout : 0.5
    get_perf_of_separate_models : 0
    grad_clip_xlmr : 0
    hidden_dim : 200
    hidden_eval : 0
    inhouse_eval : 0
    initialize_with_pretrained : 0
    input_lang : arabic
    lambda_mix : 0.8
    log_dir : logs
    log_name : eval.log.new-arabic.oregon.small-train
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
    num_epoch : 60
    num_first_xlmr_layers : 4
    num_last_layer_xlmr : 1
    optim : adam
    output_file : ./output/new-arabic.oregon.small-train.json
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
    test_file : ./datasets/8d/update2/ten_arabic_sentences.bp.v2.json
    test_is_dir : False
    train_ED : 1
    train_argument : 1
    train_file : app/train_data.bp.json
    train_is_dir : True
    train_on_arb : 0
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

if not opt['test_is_dir']:
    if opt['input_lang'] is None:
        print("Terminated! Input language is not specified")
        exit(1)
    else:
        detect_lang = read_json(opt['test_file'])
        if detect_lang['corpus-id'] == 'release-7-2019-12-23a-inclusive':
            opt['input_lang'] = 'english'

        from python.clever.event_models.uoregon.models.pipeline._01.readers import read_abstract_test_data

        # opt['test_file'] is specified on the command line
        data_map = read_abstract_test_data(opt['test_file'])
        opt['data_map'] = data_map

        from python.clever.event_models.uoregon.models.pipeline._01.trainers import *

        # ************* Run pipeline *****************
        torch.autograd.set_detect_anomaly(True)
        random.seed(opt['seed'])
        np.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed(opt['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        trainer = PipelineTrainer(opt, test_file=opt['test_file'], lang=opt['input_lang'])
        trainer.eval()
else:
    from .models.pipeline._01.readers import read_abstract_test_data

    test_files = get_files_in_dir_recur(opt['test_file'])
    output_dir = opt['output_file']
    for test_file in test_files:
        opt['test_file'] = test_file
        opt['output_file'] = os.path.join(output_dir, os.path.basename(test_file) + '.sysfile')

        detect_lang = read_json(opt['test_file'])
        if detect_lang['corpus-id'] == 'release-7-2019-12-23a-inclusive':
            opt['input_lang'] = 'english'

        data_map = read_abstract_test_data(opt['test_file'])
        opt['data_map'] = data_map

        from python.clever.event_models.uoregon.models.pipeline._01.trainers import *

        # ************* Run pipeline *****************
        torch.autograd.set_detect_anomaly(True)
        random.seed(opt['seed'])
        np.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed(opt['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        trainer = PipelineTrainer(opt, test_file=opt['test_file'], lang=opt['input_lang'])
        trainer.eval()
