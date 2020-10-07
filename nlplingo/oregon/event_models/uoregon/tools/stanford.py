import stanfordnlp, os, json, copy
from collections import defaultdict
from nlplingo.oregon.event_models.uoregon.tools.global_constants import UNK_TOKEN, PAD_TOKEN
from nlplingo.oregon.event_models.uoregon.define_opt import opt

STANFORD_RESOURCE_DIR = opt['stanford_resource_dir']	# ==>

if opt['use_ner']:
    from .stanza import Stanza_Parser

if not os.path.exists(os.path.join(STANFORD_RESOURCE_DIR, 'en_ewt_models/en_ewt_parser.pt')):
    stanfordnlp.download('en', resource_dir=STANFORD_RESOURCE_DIR, force=True)

if not os.path.exists(os.path.join(STANFORD_RESOURCE_DIR, 'ar_padt_models/ar_padt_parser.pt')):
    stanfordnlp.download('ar', resource_dir=STANFORD_RESOURCE_DIR, force=True)


class Stanford_Parser:
    def __init__(self, language='en'):
        self.lang = language
        self.pretokenized_model = stanfordnlp.Pipeline(lang=language, models_dir=STANFORD_RESOURCE_DIR,
                                                       tokenize_pretokenized=True)
        self.tokenizer = stanfordnlp.Pipeline(lang=language, models_dir=STANFORD_RESOURCE_DIR, processors='tokenize')
        if opt['use_ner']:
            self.stanza_model = Stanza_Parser(language)

    def tokenize_text(self, text):
        sent_text = copy.deepcopy(text)
        ori_text = copy.deepcopy(sent_text)
        tokens = {
            'word': [],
            'span': []
        }

        doc = self.tokenizer(sent_text)

        offset = 0
        position = 0
        for sentence in doc.sentences:
            for word in sentence.tokens:
                # ***** get span location in original text *****
                sent_text, start_char_idx = self.get_startchar_idx(word.text, sent_text)
                start_char_idx += offset
                end_char_idx = start_char_idx + len(word.text) - 1

                tokens['word'].append(ori_text[start_char_idx: end_char_idx + 1])
                tokens['span'].append((start_char_idx, end_char_idx))
                if self.lang == 'en':
                    assert tokens['word'][-1] == word.text
                offset = end_char_idx + 1
                position += 1
        pretokenized_text = ' '.join(tokens['word']).replace('\n', ' ')
        return pretokenized_text

    def get_features(self, sent_text):
        ori_text = copy.deepcopy(sent_text)
        tokens = {
            'word': [],
            'lemma': [],
            'upos': [],
            'xpos': [],
            'morph': [],
            'head': [],
            'dep_rel': [],
            'ner': [],
            'span': []
        }
        if len(sent_text.strip()) == 0:
            return tokens

        pretokenized_text = self.tokenize_text(sent_text)

        doc = self.pretokenized_model(pretokenized_text)
        if opt['use_ner']:
            stanza_tokens = self.stanza_model.get_ner_features(pretokenized_text)
        else:
            stanza_tokens = {'ner': []}
        assert len(doc.sentences) == 1

        offset = 0
        position = 0
        for sentence in doc.sentences:
            for tok in sentence.tokens:
                for word in tok.words:
                    tokens['lemma'].append(word.lemma)
                    tokens['upos'].append(word.upos)
                    tokens['xpos'].append(word.xpos)
                    tokens['morph'].append(word.feats)
                    tokens['head'].append(int(word.governor))
                    tokens['dep_rel'].append(word.dependency_relation.split(':')[0])

                    # ***** get ner info from stanza ***************
                    if len(tokens['ner']) >= len(stanza_tokens['ner']):
                        tokens['ner'].append('O')
                    else:
                        tokens['ner'].append(stanza_tokens['ner'][len(tokens['ner'])])
                    # ***** get span location in original text *****
                    sent_text, start_char_idx = self.get_startchar_idx(word.text, sent_text)
                    start_char_idx += offset
                    end_char_idx = start_char_idx + len(word.text) - 1

                    tokens['word'].append(ori_text[start_char_idx: end_char_idx + 1])
                    tokens['span'].append((start_char_idx, end_char_idx))

                    assert tokens['word'][-1] == word.text

                    offset = end_char_idx + 1
                    position += 1
        assert len(tokens['word']) == len(tokens['upos'])
        return tokens

    def get_startchar_idx(self, word, text):
        # ******* search for first non-space character *******
        start_char_idx = 0
        for k in range(len(text)):
            if len(text[k].strip()) > 0:
                start_char_idx = k
                break
        text = text[start_char_idx + len(word):]
        return text, start_char_idx


LANG_SET = {'english', 'arabic'}

stanford_parser = {
    'english': Stanford_Parser('en'),
    'arabic': Stanford_Parser('ar')
}

if __name__ == '__main__':
    text = "خرجت مجموعة تتالف من أكثر من 1000 طالب في مسيرة عبر وسط لندن في احتجاج شهد هجوماً على حافلة الشرطة.‬"
    features = stanford_parser['arabic'].get_features(text)
    print(features)
