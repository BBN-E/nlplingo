import os
from transformers import BertTokenizer
from nlplingo.oregon.event_models.uoregon.define_opt import opt
from nlplingo.oregon.event_models.uoregon.tools.global_constants import CLS_TOKEN, SEP_TOKEN, BERT_RESOURCE_DIR


class BertTool:
    def __init__(self, bert_version):
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

    def get_token_ids(self, stanford_words, trigger_word=None):
        text = '{} '.format(CLS_TOKEN) + ' '.join(stanford_words) + ' {}'.format(SEP_TOKEN)
        if trigger_word is not None:
            text += ' {} {}'.format(trigger_word, SEP_TOKEN)
        bert_words = self.tokenizer.wordpiece_tokenizer.tokenize(
            text=text
        )
        bert_tokens, stanford_retrieve_ids = self.map_wordpiece_indices(
            stanford_words=stanford_words,
            bert_words=bert_words
        )
        assert len(stanford_words) == len(stanford_retrieve_ids)
        return bert_tokens, stanford_retrieve_ids

    def map_wordpiece_indices(self, stanford_words, bert_words):
        bert_tokens = self.tokenizer.convert_tokens_to_ids(bert_words)
        retrieve_ids = []
        ori_index = 0
        bert_index = 0
        while ori_index < len(stanford_words) and bert_index < len(bert_words):
            bert_token = bert_words[bert_index]
            if bert_token.startswith('##') or bert_token.startswith(CLS_TOKEN) or bert_token.startswith(SEP_TOKEN):
                bert_index += 1
            else:
                retrieve_ids.append(bert_index)
                ori_index += 1
                bert_index += 1

        if len(retrieve_ids) != len(stanford_words):
            bert_words = [CLS_TOKEN] + stanford_words + [SEP_TOKEN]
            bert_tokens = self.tokenizer.convert_tokens_to_ids(bert_words)
            retrieve_ids = list(range(len(bert_tokens)))[1:-1]

        assert len(retrieve_ids) == len(stanford_words)
        return bert_tokens, retrieve_ids


if '-cased' in opt['bert_version']:
    cased_ver = opt['bert_version']
    uncased_ver = cased_ver.replace('-cased', '-uncased')
else:
    uncased_ver = opt['bert_version']
    cased_ver = uncased_ver.replace('-uncased', '-cased')

bert_cased = BertTool(cased_ver)
bert_uncased = BertTool(uncased_ver)

if not os.path.exists(os.path.join(BERT_RESOURCE_DIR)):
    os.mkdir(os.path.join(BERT_RESOURCE_DIR))

if not os.path.exists(os.path.join(BERT_RESOURCE_DIR, cased_ver)):
    os.mkdir(os.path.join(BERT_RESOURCE_DIR, cased_ver))

if not os.path.exists(os.path.join(BERT_RESOURCE_DIR, uncased_ver)):
    os.mkdir(os.path.join(BERT_RESOURCE_DIR, uncased_ver))
