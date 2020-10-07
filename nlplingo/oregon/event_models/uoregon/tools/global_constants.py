import os, json

WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

IMPACT_KEY = "helpful-harmful"
EFFECT_KEY = "material-verbal"

HARMFUL_KEY = 'harmful'
HELPFUL_KEY = 'helpful'
NEUTRAL_KEY = 'neutral'

MATERIAL_KEY = 'material'
VERBAL_KEY = 'verbal'
MATERIAL_VERBAL_KEY = 'both'

UNKNOWN_EVENT_KEY = 'unk'

ANNOT_SPLIT_STR = '\n{}\n'.format('=' * 10)
ENT_SPLIT_STR = '<,>'

BIO_KEY = 'BIO'

PAD_TOKEN = '[PAD]'  # consistent with BERT
UNK_TOKEN = '[UNK]'  # consistent with BERT
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
ROOT_TOKEN = '[ROOT]'  # for node whose head is root

PAD_ID = 0
UNK_ID = 1

BERT_PRETRAINED_MODEL_NAMES = {
    'bert-base-uncased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin'
    },
    'bert-large-uncased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin'
    },
    'bert-base-cased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin'
    },
    'bert-large-cased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin'
    },
    'bert-base-multilingual-uncased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin'
    },
    'bert-base-multilingual-cased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin'
    },
    'bert-base-chinese': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin'
    },
    'bert-base-german-cased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin'
    },
    'bert-large-uncased-whole-word-masking': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin'
    },
    'bert-large-cased-whole-word-masking': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin'
    },
    'bert-large-uncased-whole-word-masking-finetuned-squad': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin'
    },
    'bert-large-cased-whole-word-masking-finetuned-squad': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin'
    },
    'bert-base-cased-finetuned-mrpc': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin'
    },
    'bert-base-german-dbmdz-cased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin'
    },
    'bert-base-german-dbmdz-uncased': {
        'config-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json',
        'vocab-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt',
        'model-file': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin'
    },
}

INFINITY_NUMBER = float("inf")

MAX_BERT_TOKENS = 512
SAFE_BERT_TOKENS = MAX_BERT_TOKENS - 50
BERT_LAYERS = 13  # the first layer is for raw embedding, the next 12 layers are transformer layers

#STANFORD_RESOURCE_DIR = os.path.join(WORKING_DIR, 'tools', 'stanford_resources')	# <==
STANZA_RESOURCE_DIR = os.path.join(WORKING_DIR, 'tools', 'stanza_resources')
BERT_RESOURCE_DIR = os.path.join(WORKING_DIR, 'tools', 'bert_resources')
