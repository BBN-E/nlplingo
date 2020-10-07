INTEGERS = {'max_sentence_length', 'neighbor_distance', 'positive_weight', 'epoch', 'number_of_feature_maps',
            'position_embedding_vector_length', 'entity_embedding_vector_length',
            'batch_size', 'ner_dim', 'pos_dim', 'dep_rel_dim', 'hidden_dim', 'num_graph_cnn_layers',
            'topn', 'prune_k', 'mlp_layers', 'rnn_hidden', 'rnn_layers', 'log_step', 'save_epoch', 'patience',
            'fine_tune_epoch', 'num_batches', 'self_att_heads', 'num_first_xlmr_layers'}

LISTS = {'hidden_layers', 'cnn_filter_lengths'}

FLOATS = {'dropout', 'input_dropout', 'gcn_dropout', 'word_dropout', 'conv_l2', 'pooling_l2', 'rnn_dropout',
          'max_grad_norm', 'decoding_threshold'}

BOOLEANS = {'early_stopping', 'use_event_embedding', 'train_embeddings',
            'lower', 'no_adj', 'rnn', 'load', 'use_position_padding', 'is_embedding_vector', 'absolute_ckpt', 'decode_mode', 'continue_training', 'mention_pool'}

STRINGS = {'pooling', 'log', 'save_dir', 'exp_id', 'info', 'model_file',
           'opennre_rootpath', 'opennre_ckpt', 'opennre_dataset', 'save_model_path', 'log_dir',
           'encoder', 'dev_score_file', 'test_score_file'}


class HyperParameters(object):
    def __init__(self, params, load_from_file=False, from_pytorch=False):
        # Set the attribute of the HyperParameters object to the given value
        # e.g., if the configuration file contains 'max_sentence_length' : 128,
        # the below will set self.max_sentence_length = 128

        for integer_variable in INTEGERS:
            if integer_variable in params:
                setattr(self, integer_variable, params.get(integer_variable))

        for list_variable in LISTS:
            if list_variable in params:
                setattr(self, list_variable, params.get(list_variable))

        for float_variable in FLOATS:
            if float_variable in params:
                setattr(self, float_variable, params.get(float_variable))

        for boolean_variable in BOOLEANS:
            if boolean_variable in params:
                setattr(self, boolean_variable, params.get(boolean_variable))

        for string_variable in STRINGS:
            if string_variable in params:
                setattr(self, string_variable, params.get(string_variable))

# Arguments for graph cnn model: commented out code left here for purposes of describing what the parameter does
"""
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')

parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04,
                    help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.set_defaults(lower=False)

parser.add_argument('--prune_k', default=-1, type=int,
                    help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max',
                    help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=5, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
"""
