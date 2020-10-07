import stanza
import os, json, copy
from collections import defaultdict
from nlplingo.oregon.event_models.uoregon.tools.global_constants import UNK_TOKEN, PAD_TOKEN, STANZA_RESOURCE_DIR

if not os.path.exists(STANZA_RESOURCE_DIR):
    os.mkdir(STANZA_RESOURCE_DIR)

if not os.path.exists(os.path.join(STANZA_RESOURCE_DIR, 'en_ewt_models/en_ewt_parser.pt')):
    stanza.download('en', dir=STANZA_RESOURCE_DIR)

if not os.path.exists(os.path.join(STANZA_RESOURCE_DIR, 'ar_padt_models/ar_padt_parser.pt')):
    stanza.download('ar', dir=STANZA_RESOURCE_DIR)


class Stanza_Parser:
    def __init__(self, language='en'):
        self.lang = language
        self.pretokenized_model = stanza.Pipeline(lang=language, dir=STANZA_RESOURCE_DIR,
                                                  tokenize_pretokenized=True)

    def get_ner_features(self, pretokenized_text):
        tokens = {
            'ner': []
        }

        doc = self.pretokenized_model(pretokenized_text)

        assert len(doc.sentences) == 1

        for sentence in doc.sentences:
            for tok in sentence.tokens:
                for word in tok.words:
                    tokens['ner'].append(tok.ner)  # all words inside a token share the same tag

        return tokens
