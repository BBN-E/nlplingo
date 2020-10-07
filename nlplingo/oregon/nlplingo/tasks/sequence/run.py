from collections import defaultdict

from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.nn.hyperparameters import HyperParameters
from nlplingo.tasks.event_domain import EventDomain
from nlplingo.tasks.sequence.generator import SequenceExampleGenerator
from nlplingo.tasks.sequence.utils import get_trigger_annotations_from_docs, get_frame_annotations_from_docs

from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.define_opt import opt
from nlplingo.oregon.nlplingo.tasks.sequence.ED_trainer import EDTrainer
from nlplingo.oregon.nlplingo.tasks.sequence.argument_trainer import ArgumentTrainer
from nlplingo.oregon.nlplingo.tasks.sequence.pipeline_trainer import PipelineTrainer

# assert opt['input_lang'] == 'english'


class MyLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        with open(output_path, 'w') as f:
            f.write('-' * 50 + ' Begin logging ' + '-' * 50 + '\n')

    def info(self, string):
        with open(self.output_path, 'a') as f:
            f.write(string + '\n')


def train_trigger_from_docs(params, extractor_params, domain, hyper_params):
    """
    :type params: dict
    """
    domain.create_sequence_types(domain.event_types, use_only_begin_tag=True)

    # the following used to be command line arguments. We now get them from params file and assign accordingly
    #opt['input_lang'] = 'english'
    opt['train_ED'] = 1
    opt['train_argument'] = 0
    opt['log_name'] = 'train.log.ed'
    opt['num_epoch'] = hyper_params.epoch
    opt['docker_run'] = 0
    opt['xlmr_version'] = 'xlmr.base'
    opt['xlmr_model_dir'] = os.path.join(params['resource_dir'], 'models', 'xlmr.base')
    opt['log_dir'] = params['output_dir']
    opt['biw2v_map_dir'] = os.path.join(params['resource_dir'], 'aligned_w2v')
    opt['save_every_epoch'] = extractor_params.get('save_every_epoch', False)
    # if extractor_params.get('save_every_epoch', False):
    #     opt['delete_nonbest_ckpts'] = 0
    #     opt['save_last_epoch'] = 0
    opt['lr'] = extractor_params['optimizer']['lr']
    opt['batch_size'] = hyper_params.batch_size
    opt['self_att_heads'] = hyper_params.self_att_heads
    opt['trigger_hidden_layers'] = hyper_params.hidden_layers
    #print('hidden_layers=', opt['hidden_layers'])
    opt['trigger_model_dir'] = extractor_params['model_file']
    opt['cache_dir'] = extractor_params['cache_dir']

    logger = MyLogger(output_path=os.path.join(params['output_dir'], opt['log_name']))

    opt['train_strategy'] = 'retrain.add-all'
    #opt['observed_train'] = 'datasets/8d/update2/abstract-8d-inclusive.train.update2.bp.json'
    #opt['dev_file'] = 'datasets/8d/update2/abstract-8d-inclusive.analysis.update2.bp.json'

    """ printing out contents of opt
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
    input_lang : english
    lambda_mix : 0.8
    log_dir : logs
    log_name : train.log.ed
    lr : 2e-05
    lstm_add_satt : 0
    lstm_by_satt : 0
    lstm_layers_entity : 1
    lstm_layers_event : 1
    lstm_layers_trigger : 4
    max_grad_norm : 5.0
    mode : train_trigger_from_docs
    model : pipeline-01
    ner_dim : 30
    num_epoch : 10
    num_first_xlmr_layers : 4
    num_last_layer_xlmr : 1
    optim : adam
    output_file : None
    output_format : json
    output_offsets : 1
    params : nlplingo/trigger/params/train.params
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
    train_ED : 1
    train_argument : 0
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

    """ print(domain.sequence_types)
    {'B-harmful.both': 0, 'B-harmful.material': 1, 'B-harmful.unk': 2, 'B-harmful.verbal': 3, 'B-helpful.both': 4,
     'B-helpful.material': 5, 'B-helpful.unk': 6, 'B-helpful.verbal': 7, 'B-neutral.both': 8, 'B-neutral.material': 9,
     'B-neutral.unk': 10, 'B-neutral.verbal': 11, 'I-harmful.both': 12, 'I-harmful.material': 13, 'I-harmful.unk': 14,
     'I-harmful.verbal': 15, 'I-helpful.both': 16, 'I-helpful.material': 17, 'I-helpful.unk': 18,
     'I-helpful.verbal': 19, 'I-neutral.both': 20, 'I-neutral.material': 21, 'I-neutral.unk': 22,
     'I-neutral.verbal': 23, 'O': 24}
    """
    
    example_generator = SequenceExampleGenerator(domain, params, extractor_params, hyper_params)

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""
    train_annotations = get_trigger_annotations_from_docs(train_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""
    train_examples = example_generator.generate_spans_for_training(train_docs, train_annotations, use_only_begin_tag=True)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    dev_docs = prepare_docs(params['data']['dev']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""
    dev_annotations = get_trigger_annotations_from_docs(dev_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""
    dev_examples = example_generator.generate_spans_for_training(dev_docs, dev_annotations, use_only_begin_tag=True)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    # pos_tags = defaultdict(int)
    # for doc in train_docs + dev_docs:
    #     for sentence in doc.sentences:
    #         for token in sentence.tokens:
    #             pos_tags[token.pos_tag] += 1
    # for tag, count in sorted(pos_tags.items(), key=lambda item: item[1], reverse=True):
    #     print('%s %d' % (tag, count))
    # sys.exit(0)


    torch.autograd.set_detect_anomaly(True)
    # YS: following are for reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    biw2v_map = load_embedding_maps()

    # the following initializes the EDModel, then creates train_iterator and dev_iterator, to encode examples and creates batches
    ED_trainer = EDTrainer(opt, logger, biw2v_map, eval_mode=False, train_examples=train_examples, train_docs=train_docs,
                           dev_examples=dev_examples, dev_docs=dev_docs, label_map=domain.sequence_types)
    if opt['get_perf_of_separate_models']:
        ED_trainer.eval_with_saved_model()
    else:
        ED_trainer.train()


def train_argument_from_docs(params, extractor_params, domain, hyper_params):
    """
    :type params: dict
    """
    domain.create_sequence_types(domain.event_roles)

    continue_training = extractor_params.get('continue_training', False)
    if continue_training:
        opt['train_on_arb'] = 1
        opt['initialize_with_pretrained'] = 1
        opt['finetune_on_arb'] = 1

    if hasattr(hyper_params, 'num_first_xlmr_layers'):
        opt['num_first_xlmr_layers'] = hyper_params.num_first_xlmr_layers
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.embed_tokens')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.embed_positions')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.embed_tokens')
        opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.embed_positions')
        opt['finetuned_xlmr_layers'].append('self_att.attention_layers')
        opt['finetuned_xlmr_layers'].append('gcn_layer')
        opt['finetuned_xlmr_layers'].append('biw2v_embedding')
        for k in range(opt['num_first_xlmr_layers']):
            opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.decoder.sentence_encoder.layers.{}.'.format(k))
            opt['finetuned_xlmr_layers'].append('xlmr_embedding.model.encoder.sentence_encoder.layers.{}.'.format(k))

    # the following used to be command line arguments. We now get them from params file and assign accordingly
    #opt['input_lang'] = 'english'
    opt['output_dir'] = params['output_dir']
    opt['train_ED'] = 0
    opt['train_argument'] = 1
    opt['log_name'] = 'train.log.arg'
    opt['num_epoch'] = hyper_params.epoch
    opt['docker_run'] = 0
    opt['xlmr_version'] = 'xlmr.base'
    opt['xlmr_model_dir'] = os.path.join(params['resource_dir'], 'models', 'xlmr.base')
    opt['log_dir'] = params['output_dir']
    opt['biw2v_map_dir'] = os.path.join(params['resource_dir'], 'aligned_w2v')
    opt['save_every_epoch'] = extractor_params.get('save_every_epoch', False)
    # if extractor_params.get('save_every_epoch', False):
    #     opt['delete_nonbest_ckpts'] = 0
    #     opt['save_last_epoch'] = 0
    opt['lr'] = extractor_params['optimizer']['lr']
    opt['batch_size'] = hyper_params.batch_size
    opt['self_att_heads'] = hyper_params.self_att_heads
    opt['argument_hidden_layers'] = hyper_params.hidden_layers
    #print('hidden_layers=', opt['hidden_layers'])
    opt['argument_model_dir'] = extractor_params['model_file']

    logger = MyLogger(output_path=os.path.join(params['output_dir'], opt['log_name']))

    example_generator = SequenceExampleGenerator(domain, params, extractor_params, hyper_params)

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""
    train_annotations = get_frame_annotations_from_docs(train_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""
    train_examples = example_generator.generate_frames_for_training(train_docs, train_annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    dev_docs = prepare_docs(params['data']['dev']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""
    dev_annotations = get_frame_annotations_from_docs(dev_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""
    dev_examples = example_generator.generate_frames_for_training(dev_docs, dev_annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    torch.autograd.set_detect_anomaly(True)
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    biw2v_map = load_embedding_maps()

    arg_trainer = ArgumentTrainer(opt, logger, biw2v_map, eval_mode=False, train_examples=train_examples, train_docs=train_docs,
                                  dev_examples=dev_examples, dev_docs=dev_docs, label_map=domain.sequence_types)
    if opt['get_perf_of_separate_models']:
        arg_trainer.eval_with_saved_model()
    else:
        arg_trainer.train()

def decode(params, trigger_extractor_params, argument_extractor_params, trigger_domain, argument_domain, trigger_hyper_params, argument_hyper_params):
#def eval(params, extractor_params, hyper_params, trigger_domain, argument_domain):
    # the following used to be command line arguments. We now get them from params file and assign accordingly
    trigger_domain.create_sequence_types(trigger_domain.event_types, use_only_begin_tag=True)
    argument_domain.create_sequence_types(argument_domain.event_roles)

    #opt['input_lang'] = params['lang']
    opt['output_offsets'] = 1

    ##opt['test_file'] = params['bp_file']  # BP-JSON file TODO should be made optional
    ##opt['output_file'] = params['output_bp']

    #opt['train_ED'] = 0
    #opt['train_argument'] = 1
    opt['log_name'] = 'decode.log'
    #opt['num_epoch'] = hyper_params.epoch
    opt['docker_run'] = 0
    #opt['xlmr_version'] = 'xlmr.base'
    opt['xlmr_model_dir'] = os.path.join(params['resource_dir'], 'models', 'xlmr.base')
    opt['log_dir'] = params['output_dir']
    opt['biw2v_map_dir'] = os.path.join(params['resource_dir'], 'aligned_w2v')
    #if extractor_params.get('save_every_epoch', False):
    #    opt['delete_nonbest_ckpts'] = 0
    #    opt['save_last_epoch'] = 0
    #opt['lr'] = extractor_params['optimizer']['lr']
    #opt['batch_size'] = hyper_params.batch_size

    opt['trigger_model_dir'] = trigger_extractor_params['model_file']
    opt['argument_model_dir'] = argument_extractor_params['model_file']

    opt['trigger_hidden_layers'] = trigger_hyper_params.hidden_layers
    opt['argument_hidden_layers'] = argument_hyper_params.hidden_layers

    logger = MyLogger(output_path=os.path.join(params['output_dir'], opt['log_name']))

    example_generator = SequenceExampleGenerator(trigger_domain, params, trigger_extractor_params, trigger_hyper_params)

    test_docs = prepare_docs(params['data']['test']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    opt['decode_with_gold_triggers'] = argument_extractor_params.get('decode_with_gold_triggers', False)
    if opt['decode_with_gold_triggers']:
        annotations = get_trigger_annotations_from_docs(test_docs)
        """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""
        examples = example_generator.generate_spans_for_training(test_docs, annotations, use_only_begin_tag=True)
        """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""
    else:
        examples = example_generator.generate_spans_for_decoding(test_docs)
        """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    torch.autograd.set_detect_anomaly(True)
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    biw2v_map = load_embedding_maps()

    opt['predictions_file'] = params['predictions_file']

    # params['test_file'] should point to a BP-JSON
    trainer = PipelineTrainer(opt, logger, biw2v_map, examples, test_docs, trigger_domain.sequence_types, argument_domain.sequence_types)
    trainer.eval()


def load_embedding_maps():
    # with open(os.path.join(WORKING_DIR, 'tools', 'aligned_w2v/biw2v.vocab.txt')) as f:		# <==
    #    biw2v_vocab = [line.strip() for line in f.readlines() if len(line.strip()) > 0]   		# <==
    # biw2v_vecs = np.load(os.path.join(WORKING_DIR, 'tools', 'aligned_w2v/biw2v.embed.npy'))		# <==

    with open(os.path.join(opt['biw2v_map_dir'], 'biw2v.vocab.txt')) as f:		  		# ==>
        biw2v_vocab = [line.strip() for line in f.readlines() if len(line.strip()) > 0]			# ==>
    biw2v_vecs = np.load(os.path.join(opt['biw2v_map_dir'], 'biw2v.embed.npy'))				# ==>
    opt['biw2v_vecs'] = biw2v_vecs
    opt['biw2v_size'] = len(biw2v_vocab)

    biw2v_map = {}
    for word_id, word in enumerate(biw2v_vocab):
        biw2v_map[word] = word_id
    return biw2v_map


if __name__ == "__main__":
    #logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    #parser.add_argument('--params', required=True)

    #args = parser.parse_args()

    with open(opt['params']) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    trigger_hyper_params = None
    argument_hyper_params = None
    trigger_domain = None
    argument_domain = None
    trigger_extractors = []
    argument_extractors = []
    for extractor_params in params['extractors']:
        if extractor_params['task'] == 'event-trigger':
            trigger_extractors.append(extractor_params)
            trigger_domain = EventDomain.read_domain_ontology_file(extractor_params['domain_ontology'], domain_name=extractor_params.get('domain_name', 'general'))
            #trigger_domain.create_sequence_types(trigger_domain.event_types, use_only_begin_tag=True)
            trigger_hyper_params = HyperParameters(extractor_params['hyper-parameters'], load_from_file=False)
        elif extractor_params['task'] == 'event-argument':
            argument_extractors.append(extractor_params)
            argument_domain = EventDomain.read_domain_ontology_file(extractor_params['domain_ontology'], domain_name=extractor_params.get('domain_name', 'general'))
            #argument_domain.create_sequence_types(argument_domain.event_roles)
            argument_hyper_params = HyperParameters(extractor_params['hyper-parameters'], load_from_file=False)

    if opt['mode'] == 'train_trigger_from_file':
        train_trigger_from_docs(params, trigger_extractors[0], trigger_domain, trigger_hyper_params)
    elif opt['mode'] == 'train_argument':
        train_argument_from_docs(params, argument_extractors[0], argument_domain, argument_hyper_params)
    elif opt['mode'] == 'decode_trigger_argument':
        decode(params, trigger_extractors[0], argument_extractors[0], trigger_domain, argument_domain, trigger_hyper_params, argument_hyper_params)
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(opt['mode']))
