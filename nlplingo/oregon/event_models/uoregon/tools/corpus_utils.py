import json, os
import numpy as np
from datetime import datetime
from nlplingo.oregon.event_models.uoregon.define_opt import opt
#from .stanford import LANG_SET, stanford_parser
import unicodedata, re, copy

__FORMAT_TYPE__ = "bp-corpus"
__FORMAT_VERSION__ = "v8f"

LANG_SET = {'english', 'arabic'}


def strip_punctuations(text):
    stripped = re.sub(
        r'''^[ "’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]+|[ "’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]+$''',
        '', text).strip('\n')
    if len(stripped) == 0:
        return '', (-1, -1)
    else:
        start = 0
        while start < len(text) and text[start] != stripped[0]:
            start += 1

        end = -1
        while end >= - len(text) and text[end] != stripped[-1]:
            end -= 1
        end += 1
        return stripped, (start, -end)


def normalize_text(text, do_mapping=False):
    def normalize(text):
        norm_text = unicodedata.normalize('NFKC', text)
        norm_text = re.sub(
            r'''[^\w\s"’'\(\)\[\]\{\}<>:\,‒–—―!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#%‰‱¶′§~_\|¦⁂☞∴‽※"]''', '',
            norm_text)
        return norm_text

    def find_minmatching(norm_text, ori_text, i, j):
        min_sum = 1000000
        min_k, min_h = -1, -1
        for k in range(i, len(norm_text)):
            for h in range(j, len(ori_text)):
                if norm_text[k] == normalize(ori_text[h]) and k + h < min_sum:
                    min_sum = k + h
                    min_k = k
                    min_h = h
        return min_k, min_h

    if opt['test_file'] is not None and opt['input_lang'] == 'arabic' and os.path.basename(
            opt['test_file']) != 'arabic-abstract-sample.bp.json':
        ori_text = copy.deepcopy(text)
        norm_text = normalize(text)
        # ******* map: norm_text -> text **********
        norm2ori_offsetmap = {}
        if do_mapping:
            i, j = 0, 0
            while i < len(norm_text) and j < len(ori_text):
                n_i = norm_text[i]
                o_j = normalize(ori_text[j])
                if len(o_j) == 0:
                    j += 1
                else:
                    if n_i == o_j:
                        norm2ori_offsetmap[i] = j
                        i += 1
                        j += 1
                    else:
                        min_k, min_h = find_minmatching(norm_text, ori_text, i, j)
                        for t in range(i, min_k):
                            norm2ori_offsetmap[t] = min_h - 1
                        i, j = min_k, min_h

            assert len(norm2ori_offsetmap) == len(norm_text)
        return norm_text, ori_text, norm2ori_offsetmap
    else:
        norm2ori_offsetmap = dict([(k, k) for k in range(len(text))])
        return text, text, norm2ori_offsetmap


def get_ori_string(ori_example, norm_offset):
    norm2ori_offsetmap = ori_example['norm2ori_offsetmap']
    ori_text = ori_example['ori_text']
    norm_text = ori_example['text']
    norm_string = norm_text[norm_offset[0]: norm_offset[-1]]

    if opt['test_file'] is not None and opt['input_lang'] == 'arabic' and os.path.basename(
            opt['test_file']) != 'arabic-abstract-sample.bp.json':
        stripped_string, (start, end) = strip_punctuations(norm_string)
    else:
        stripped_string = norm_string
        start, end = 0, 0
    if len(stripped_string) == 0:
        return [norm_offset[0], norm_offset[0]], ''
    else:
        norm_offset = [norm_offset[0] + start, norm_offset[1] - end]

        ori_offset = [norm2ori_offsetmap[norm_offset[0]], norm2ori_offsetmap[norm_offset[-1] - 1] + 1]
        ori_string = ori_text[ori_offset[0]: ori_offset[-1]]
        return ori_offset, ori_string


class Corpus:
    def __init__(self, filepath, lang, parsing=True, normalize=True):
        self.__init_time = datetime.now()
        if not os.path.exists(filepath):
            print('Terminated! {} doesnt exist'.format(filepath))
            #logger.info('Terminated! {} doesnt exist'.format(filepath))
            exit(1)
        with open(filepath, 'r', encoding='utf8') as f:
            data = json.load(f)
        assert lang in LANG_SET, 'Unsupported language! Supported languages: {}'.format(LANG_SET)
        self.__lang = lang
        self.__corpus_id = data['corpus-id']
        self.__format_type = data['format-type']
        self.__format_version = data['format-version']
        # self.__provenance = data['provenance']	# <==
        self.__provenance = {}				# ==>
        self.__parsing = parsing

        assert (self.__format_type == __FORMAT_TYPE__)
        # assert (self.__format_version == __FORMAT_VERSION__)
        self.__docs = dict()

        for entry_id, entry_value in data['entries'].items():
            assert (entry_id == entry_value['entry-id'])
            doc_id = entry_value['doc-id']
            if doc_id not in self.__docs:
                self.__docs[doc_id] = Document(doc_id, lang=lang, parsing=self.__parsing)
            self.__docs[doc_id].add_entry(entry_value, normalize)

        self.__sentences = []

        for doc in self.__docs.values():
            doc.sort_sentences()
            self.__sentences += doc.sentences

        self.__eid2sent = dict()
        for sent in self.__sentences:
            self.__eid2sent[sent.entry_id] = sent
        print(
            'Parsing corpus {} is done!\nParsing time: {} - {}\nLanguage: {}\n#entries: {}\n'.format(
                os.path.basename(filepath),
                self.__init_time,
                datetime.now(), self.lang,
                len(self.__sentences)) +
            '#abstract-events: {}\n#helpful: {}\n#harmful: {}\n#neutral: {}\n#unk: {}\n'.format(
                np.sum([len(sent.abstract_events) for sent in self.__sentences]),
                np.sum([len(
                    [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'helpful'])
                        for sent in self.__sentences]),
                np.sum([len(
                    [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'harmful'])
                        for sent in self.__sentences]),
                np.sum([len(
                    [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'neutral'])
                        for sent in self.__sentences]),
                np.sum(
                    [len([event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'unk'])
                     for sent in self.__sentences]),
            ) +
            '#material: {}\n#verbal: {}\n#both: {}\n#unk: {}\n'.format(
                np.sum([len(
                    [event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'material'])
                        for sent in self.__sentences]),
                np.sum([len(
                    [event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'verbal'])
                        for sent in self.__sentences]),
                np.sum(
                    [len([event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'both'])
                     for sent in self.__sentences]),
                np.sum(
                    [len([event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'unk'])
                     for sent in self.__sentences]),
            )
        )
        # logger.info(
        #     'Parsing corpus {} is done!\nParsing time: {} - {}\nLanguage: {}\n#entries: {}\n'.format(
        #         os.path.basename(filepath),
        #         self.__init_time,
        #         datetime.now(), self.lang,
        #         len(self.__sentences)) +
        #     '#abstract-events: {}\n#helpful: {}\n#harmful: {}\n#neutral: {}\n#unk: {}\n'.format(
        #         np.sum([len(sent.abstract_events) for sent in self.__sentences]),
        #         np.sum([len(
        #             [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'helpful'])
        #                 for sent in self.__sentences]),
        #         np.sum([len(
        #             [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'harmful'])
        #                 for sent in self.__sentences]),
        #         np.sum([len(
        #             [event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'neutral'])
        #                 for sent in self.__sentences]),
        #         np.sum(
        #             [len([event for event_id, event in sent.abstract_events.items() if event.helpful_harmful == 'unk'])
        #              for sent in self.__sentences]),
        #     ) +
        #     '#material: {}\n#verbal: {}\n#both: {}\n#unk: {}\n'.format(
        #         np.sum([len(
        #             [event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'material'])
        #                 for sent in self.__sentences]),
        #         np.sum([len(
        #             [event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'verbal'])
        #                 for sent in self.__sentences]),
        #         np.sum(
        #             [len([event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'both'])
        #              for sent in self.__sentences]),
        #         np.sum(
        #             [len([event for event_id, event in sent.abstract_events.items() if event.material_verbal == 'unk'])
        #              for sent in self.__sentences]),
        #     )
        # )

    @property
    def lang(self):
        return self.__lang

    @property
    def corpus_id(self):
        return self.__corpus_id

    @property
    def format_type(self):
        return self.__format_type

    @property
    def format_version(self):
        return self.__format_version

    @property
    def provenance(self):
        return self.__provenance

    @property
    def docs(self):
        """
        Returns:
            dict
        """
        return self.__docs

    @property
    def sentences(self):
        return self.__sentences

    @property
    def eid2sent(self):
        return self.__eid2sent

    def clear_annotation(self):
        for doc in self.docs.values():
            doc.clear_annotation()

    def save(self, output_file):
        entries = {}
        for _, doc in self.docs.items():
            doc_entries = doc.to_json_dict()
            for entry_id, entry_value in doc_entries.items():
                assert (entry_id not in entries)
                entries[entry_id] = entry_value # YS: entry_value is Sentence.to_json_dict
        data = {
            'corpus-id': self.__corpus_id,
            'entries': entries,     # YS: these are Sentence.to_json_dict
            'format-type': self.__format_type,
            'format-version': self.__format_version,
            'provenance': self.__provenance
        }
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(
                data, output, ensure_ascii=False, indent=2, sort_keys=True)


class Document:
    def __init__(self, doc_id, lang, parsing):
        assert lang in LANG_SET
        self.__lang = lang
        self.__doc_id = doc_id
        self.__sentences = []
        self.__parsing = parsing

    @property
    def lang(self):
        return self.__lang

    @property
    def doc_id(self):
        return self.__doc_id

    @property
    def sentences(self):
        return self.__sentences

    def add_entry(self, entry_dict, normalize=True):
        if entry_dict['segment-type'] == 'sentence':
            sentence = Sentence(doc_id=self.doc_id, entry_dict=entry_dict, lang=self.lang, parsing=self.__parsing,
                                normalize=normalize)
            self.__sentences.append(sentence)
        else:
            raise RuntimeError(
                'segment-type: {} not implemented!'.format(
                    entry_dict['segment-type']
                )
            )

    def sort_sentences(self):
        self.__sentences = sorted(self.__sentences, key=lambda x: x.sent_id)

    @property
    def text(self):
        out_text = ''
        for sentence in self.__sentences:
            out_text += sentence.text + "\n"
        return out_text

    def __repr__(self):
        return self.text

    def clear_annotation(self):
        for sentence in self.sentences:
            sentence.clear_annotation()

    def to_json_dict(self):
        entries = {}
        for sentence in self.sentences:
            entries[sentence.entry_id] = sentence.to_json_dict()
        return entries


class Sentence:
    def __init__(self, *, doc_id, entry_dict, lang, parsing, normalize=True):
        assert (entry_dict['segment-type'] == 'sentence')
        assert lang in LANG_SET
        self.__lang = lang
        self.__abstract_events = dict()
        self.__span_sets = dict()
        self.__doc_id = doc_id

        # YS: normalizing their own Arabic sentences
        self.__text, self.__ori_text, self.__norm2ori_offsetmap = normalize_text(entry_dict['segment-text'],
                                                                                 do_mapping=True) if normalize else (
            entry_dict['segment-text'], entry_dict['segment-text'], {})
        self.__sent_id = int(entry_dict['sent-id'])
        self.__entry_id = entry_dict['entry-id']
        self.__features = stanford_parser[lang].get_features(sent_text=self.__text) if parsing else {}

        """
        upos: universal part of speech: https://universaldependencies.org/u/pos/
        xpos: treebank specific part of speech
        morph: universal features: https://universaldependencies.org/u/feat/index.html
        head: indexing starts from 1. For each token, stores index of head. Value of 0 means this token is root
        
        self.__text= On the afternoon of Oct. 7, 1974, a mob of 200 enraged whites, many of them students, closed in on a bus filled with black students that was trying to pull away from the local high school.
        self.__features = {
                      0      1         2        3     4       5    6     7      8    9    10     11    12       13        14       15   16      17    18       19        20     21       22    23   24    25      26       27       28       29          30     31      32       33     34     35       36     37      38       39      40       41
         'word':    ['On', 'the', 'afternoon', 'of', 'Oct.', '7', ',', '1974', ',', 'a', 'mob', 'of', '200', 'enraged', 'whites', ',', 'many', 'of', 'them', 'students', ',', 'closed', 'in', 'on', 'a', 'bus', 'filled', 'with', 'black', 'students', 'that', 'was', 'trying', 'to', 'pull', 'away', 'from', 'the', 'local', 'high', 'school', '.'], 
         'head':    [ 3,    3,     14,          5,    3,      5,   5,   5,      14,  11,  14,    13,   11,    0,         14,       15,  15,     19,   17,     17,         22,  14,       22,   26,   26,  22,    26,       30,     30,      27,         33,     33,    30,       35,   33,     35,     41,     41,    41,      41,     35,       14], 
         'dep_rel': ['case', 'det', 'obl',    'case', 'nmod', 'nummod', 'punct', 'nummod', 'punct', 'det', 'nsubj', 'case', 'nmod', 'root', 'obj', 'punct', 'appos', 'case', 'nmod', 'nmod', 'punct', 'conj', 'advmod', 'case', 'det', 'obl', 'acl', 'case', 'amod', 'obl', 'nsubj', 'aux', 'acl', 'mark', 'xcomp', 'compound', 'case', 'det', 'amod', 'amod', 'obl', 'punct'], 
         'lemma':   ['on', 'the', 'afternoon', 'of', 'Oct.', '7', ',', '1974', ',', 'a', 'mob', 'of', '200', 'enrage', 'white', ',', 'many', 'of', 'they', 'student', ',', 'close', 'in', 'on', 'a', 'bus', 'fill', 'with', 'black', 'student', 'that', 'be', 'try', 'to', 'pull', 'away', 'from', 'the', 'local', 'high', 'school', '.'], 
         'upos':    ['ADP', 'DET', 'NOUN', 'ADP', 'PROPN', 'NUM', 'PUNCT', 'NUM', 'PUNCT', 'DET', 'NOUN', 'ADP', 'NUM', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'ADP', 'PRON', 'NOUN', 'PUNCT', 'VERB', 'ADV', 'ADP', 'DET', 'NOUN', 'VERB', 'ADP', 'ADJ', 'NOUN', 'PRON', 'AUX', 'VERB', 'PART', 'VERB', 'ADP', 'ADP', 'DET', 'ADJ', 'ADJ', 'NOUN', 'PUNCT'], 
         'xpos':    ['IN', 'DT', 'NN', 'IN', 'NNP', 'CD', ',', 'CD', ',', 'DT', 'NN', 'IN', 'CD', 'VBD', 'NNS', ',', 'JJ', 'IN', 'PRP', 'NNS', ',', 'VBD', 'RB', 'IN', 'DT', 'NN', 'VBN', 'IN', 'JJ', 'NNS', 'WDT', 'VBD', 'VBG', 'TO', 'VB', 'RP', 'IN', 'DT', 'JJ', 'JJ', 'NN', '.'], 
         'morph':   ['_', 'Definite=Def|PronType=Art', 'Number=Sing', '_', 'Number=Sing', 'NumType=Card', '_', 'NumType=Card', '_', 'Definite=Ind|PronType=Art', 'Number=Sing', '_', 'NumType=Card', 'Tense=Past|VerbForm=Fin', 'Number=Plur', '_', 'Degree=Pos', '_', 'Case=Acc|Number=Plur|Person=3|PronType=Prs', 'Number=Plur', '_', 'Mood=Ind|Tense=Past|VerbForm=Fin', '_', '_', 'Definite=Ind|PronType=Art', 'Number=Sing', 'Tense=Past|VerbForm=Part', '_', 'Degree=Pos', 'Number=Plur', 'PronType=Rel', 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', 'Tense=Pres|VerbForm=Part', '_', 'VerbForm=Inf', '_', '_', 'Definite=Def|PronType=Art', 'Degree=Pos', 'Degree=Pos', 'Number=Sing', '_'], 
         'ner':     ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
         'span':    [(0, 1), (3, 5), (7, 15), (17, 18), (20, 23), (25, 25), (26, 26), (28, 31), (32, 32), (34, 34), (36, 38), (40, 41), (43, 45), (47, 53), (55, 60), (61, 61), (63, 66), (68, 69), (71, 74), (76, 83), (84, 84), (86, 91), (93, 94), (96, 97), (99, 99), (101, 103), (105, 110), (112, 115), (117, 121), (123, 130), (132, 135), (137, 139), (141, 146), (148, 149), (151, 154), (156, 159), (161, 164), (166, 168), (170, 174), (176, 179), (181, 186), (187, 187)]
         }
/pytorch/aten/src/ATen/native/cuda/LegacyDefinitions.cpp:19: UserWarning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead.

        Example from another Sentence:
        self.__text= No weapon was found, and there was no evidence to indicate that the shot had come from the bus. The bus driver insisted it had not come from the bus, but from someone firing at the bus.
        self.__features = {
        'word': ['No', 'weapon', 'was', 'found', ',', 'and', 'there', 'was', 'no', 'evidence', 'to', 'indicate', 'that', 'the', 'shot', 'had', 'come', 'from', 'the', 'bus', '.', 'The', 'bus', 'driver', 'insisted', 'it', 'had', 'not', 'come', 'from', 'the', 'bus', ',', 'but', 'from', 'someone', 'firing', 'at', 'the', 'bus', '.'], 
        'lemma': ['no', 'weapon', 'be', 'find', ',', 'and', 'there', 'be', 'no', 'evidence', 'to', 'indicate', 'that', 'the', 'shot', 'have', 'come', 'from', 'the', 'bus', '.', 'the', 'bus', 'driver', 'insist', 'it', 'have', 'not', 'come', 'from', 'the', 'bus', ',', 'but', 'from', 'someone', 'fire', 'at', 'the', 'bus', '.'], 
        'upos': ['DET', 'NOUN', 'AUX', 'VERB', 'PUNCT', 'CCONJ', 'PRON', 'VERB', 'DET', 'NOUN', 'PART', 'VERB', 'SCONJ', 'DET', 'NOUN', 'AUX', 'VERB', 'ADP', 'DET', 'NOUN', 'PUNCT', 'DET', 'NOUN', 'NOUN', 'VERB', 'PRON', 'AUX', 'PART', 'VERB', 'ADP', 'DET', 'NOUN', 'PUNCT', 'CCONJ', 'ADP', 'PRON', 'VERB', 'ADP', 'DET', 'NOUN', 'PUNCT'], 
        'xpos': ['DT', 'NN', 'VBD', 'VBN', ',', 'CC', 'EX', 'VBD', 'DT', 'NN', 'TO', 'VB', 'IN', 'DT', 'NN', 'VBD', 'VBN', 'IN', 'DT', 'NN', '.', 'DT', 'NN', 'NN', 'VBD', 'PRP', 'VBD', 'RB', 'VBN', 'IN', 'DT', 'NN', ',', 'CC', 'IN', 'NN', 'VBG', 'IN', 'DT', 'NN', '.'], 
        'morph': ['_', 'Number=Sing', 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', 'Tense=Past|VerbForm=Part|Voice=Pass', '_', '_', '_', 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', '_', 'Number=Sing', '_', 'VerbForm=Inf', '_', 'Definite=Def|PronType=Art', 'Number=Sing', 'Mood=Ind|Tense=Past|VerbForm=Fin', 'Tense=Past|VerbForm=Part', '_', 'Definite=Def|PronType=Art', 'Number=Sing', '_', 'Definite=Def|PronType=Art', 'Number=Sing', 'Number=Sing', 'Mood=Ind|Tense=Past|VerbForm=Fin', 'Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs', 'Mood=Ind|Tense=Past|VerbForm=Fin', '_', 'Tense=Past|VerbForm=Part', '_', 'Definite=Def|PronType=Art', 'Number=Sing', '_', '_', '_', 'Number=Sing', 'VerbForm=Ger', '_', 'Definite=Def|PronType=Art', 'Number=Sing', '_'], 
        'head': [2, 4, 4, 0, 8, 8, 8, 4, 10, 8, 12, 10, 17, 15, 17, 17, 12, 20, 20, 17, 25, 24, 24, 25, 4, 29, 29, 29, 25, 32, 32, 29, 36, 36, 36, 29, 36, 40, 40, 37, 4], 
        'dep_rel': ['det', 'nsubj', 'aux', 'root', 'punct', 'cc', 'expl', 'conj', 'det', 'nsubj', 'mark', 'acl', 'mark', 'det', 'nsubj', 'aux', 'ccomp', 'case', 'det', 'obl', 'punct', 'det', 'compound', 'nsubj', 'parataxis', 'nsubj', 'aux', 'advmod', 'ccomp', 'case', 'det', 'obl', 'punct', 'cc', 'case', 'conj', 'acl', 'case', 'det', 'obl', 'punct'], 
        'ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
        'span': [(0, 1), (3, 8), (10, 12), (14, 18), (19, 19), (21, 23), (25, 29), (31, 33), (35, 36), (38, 45), (47, 48), (50, 57), (59, 62), (64, 66), (68, 71), (73, 75), (77, 80), (82, 85), (87, 89), (91, 93), (94, 94), (96, 98), (100, 102), (104, 109), (111, 118), (120, 121), (123, 125), (127, 129), (131, 134), (136, 139), (141, 143), (145, 147), (148, 148), (150, 152), (154, 157), (159, 165), (167, 172), (174, 175), (177, 179), (181, 183), (184, 184)]
        }
        """

        self.__wa_annotations = {} if 'wa-annotations' not in entry_dict else entry_dict['wa-annotations']

        # Right now we are assuming all entries have abstract-events
        if 'annotation-sets' in entry_dict and 'abstract-events' in entry_dict['annotation-sets']:
            abstract_events_data = entry_dict['annotation-sets']['abstract-events']
        else:
            abstract_events_data = {}

        events_data = abstract_events_data.get('events', {})
        spans_sets_data = abstract_events_data.get('span-sets', {})
        for span_set_name, span_set_value in spans_sets_data.items():
            spans = []
            for span_data in span_set_value.get('spans', []):
                spans.append(Span(span_data['string']))
            self.__span_sets[span_set_name] = SpanSet(
                span_set_name=span_set_name,
                spans=spans
            )
        for event_name, event_dict in events_data.items():
            agents = []
            patients = []
            for agent_span_set_id in event_dict['agents']:
                agents.append(self.span_sets[agent_span_set_id])
            for patient_span_set_id in event_dict['patients']:
                patients.append(self.span_sets[patient_span_set_id])
            abstract_event = AbstractEvent(
                event_id=event_dict['eventid'],
                helpful_harmful=event_dict['helpful-harmful'],
                material_verbal=event_dict['material-verbal'],
                anchor_span_set=self.span_sets[event_dict['anchors']],
                agent_span_sets=agents,
                patient_span_sets=patients,
                anchor_offsets=event_dict['anchor_offsets'] if 'anchor_offsets' in event_dict else {},
                agent_offsets=event_dict['agent_offsets'] if 'agent_offsets' in event_dict else {},
                patient_offsets=event_dict['patient_offsets'] if 'patient_offsets' in event_dict else {},
                en_ver=event_dict['en_ver'] if 'en_ver' in event_dict else {},
                event_prob=event_dict['event_prob'] if 'event_prob' in event_dict else None,
                argument_prob=event_dict['argument_prob'] if 'argument_prob' in event_dict else None
            )
            self.add_abstract_event(abstract_event)

    @property
    def lang(self):
        return self.__lang

    @property
    def abstract_events(self):
        return self.__abstract_events

    @property
    def span_sets(self):
        return self.__span_sets

    @property
    def text(self):
        return self.__text

    @property
    def ori_text(self):
        return self.__ori_text

    @property
    def norm2ori_offsetmap(self):
        return self.__norm2ori_offsetmap

    @property
    def doc_id(self):
        return self.__doc_id

    @property
    def sent_id(self):
        return self.__sent_id

    @property
    def entry_id(self):
        return self.__entry_id

    @property
    def features(self):
        return self.__features

    @property
    def wa_annotations(self):
        return self.__wa_annotations

    # Creates a span set and returns the span set id.  If an identical span set
    # already existed, that span set id is returned instead of creating a new
    # one.
    # YS: when recording predictions
    def add_span_set(self, *, span_strings):
        spans = []
        for span_string in span_strings:
            assert (span_string in self.ori_text)
            spans.append(Span(span_string))
        for ss_id, span_set in self.span_sets.items():
            if spans == span_set.spans:
                return ss_id
        new_ss_id = f'ss-{len(self.span_sets) + 1}'
        self.span_sets[new_ss_id] = SpanSet(span_set_name=new_ss_id,
                                            spans=spans)
        return new_ss_id

    # Add a new abstract event that references span sets that already exist on
    # this object
    def add_abstract_event(self, abstract_event):
        # We have to cast to string because MITRE was mixing strings and ints
        key = str(abstract_event.event_id)
        assert (key not in self.abstract_events)
        self.__abstract_events[key] = abstract_event

    def clear_annotation(self):
        self.abstract_events.clear()
        self.span_sets.clear()

    def to_json_dict(self):
        events = {}
        span_sets = {}
        for event_id, event in self.abstract_events.items():
            events[event_id] = event.to_json_dict()
        for ss_id, span_set in self.span_sets.items():
            span_sets[ss_id] = span_set.to_json_dict()
        abstract_events = {
            'events': events,
            'span-sets': span_sets
        }
        annotation_sets = {
            'abstract-events': abstract_events
        }
        sent_id = int(self.sent_id) if opt['test_file'] is not None and 'bbn' in opt['test_file'] else str(self.sent_id)
        data = {
            'annotation-sets': annotation_sets,
            'doc-id': self.__doc_id,
            'entry-id': self.entry_id,
            'segment-text': self.ori_text,
            'segment-type': 'sentence',
            'sent-id': sent_id
        }
        if len(self.wa_annotations) > 0:
            data['wa-annotations'] = self.wa_annotations
        return data


class AbstractEvent:
    # Removed SPECIFIED and NOT as they no longer show up as of 8d
    HELPFUL_HARMFUL_TYPES = {'helpful', 'harmful', 'neutral', 'unk'}
    MATERIAL_VERBAL_TYPES = {'material', 'verbal', 'both', 'unk'}

    def __init__(self, *, event_id, helpful_harmful, material_verbal,
                 anchor_span_set, agent_span_sets, patient_span_sets, anchor_offsets={}, agent_offsets={},
                 patient_offsets={}, event_prob=None, argument_prob=None, en_ver={}):
        if helpful_harmful not in self.HELPFUL_HARMFUL_TYPES:
            raise RuntimeError(
                f'Unexpected  helpful-harmful value: "{helpful_harmful}"')
        if material_verbal not in self.MATERIAL_VERBAL_TYPES:
            raise RuntimeError(
                f'Unexpected  material-verbal value: "{material_verbal}"')
        self.__event_id = event_id
        self.__helpful_harmful = helpful_harmful
        self.__material_verbal = material_verbal
        self.__anchors = anchor_span_set
        self.__agents = agent_span_sets
        self.__patients = patient_span_sets
        self.__anchor_offsets = anchor_offsets
        self.__agent_offsets = agent_offsets
        self.__patient_offsets = patient_offsets
        # self.__agent_offsets[agent_ss_id][k] corresponds to self.__agents.spans[k].string
        # self.__patient_offsets[patient_ss_id][k] corresponds to self.__patients.spans[k].string
        self.__event_prob = event_prob
        self.__argument_prob = argument_prob
        self.__en_ver = en_ver

    @property
    def agents(self):
        return self.__agents

    @property
    def patients(self):
        return self.__patients

    @property
    def agent_offsets(self):
        return self.__agent_offsets

    @property
    def patient_offsets(self):
        return self.__patient_offsets

    @property
    def event_prob(self):
        return self.__event_prob

    @property
    def argument_prob(self):
        return self.__argument_prob

    @property
    def anchors(self):
        return self.__anchors

    @property
    def anchor_offsets(self):
        return self.__anchor_offsets

    @property
    def helpful_harmful(self):
        return self.__helpful_harmful

    @property
    def material_verbal(self):
        return self.__material_verbal

    @property
    def event_id(self):
        return self.__event_id

    def to_json_dict(self):
        data = {
            'agents': sorted([x.ss_id for x in self.agents]),
            'anchors': self.anchors.ss_id,
            'eventid': self.event_id,
            'helpful-harmful': self.helpful_harmful,
            'material-verbal': self.material_verbal,
            'patients': sorted([x.ss_id for x in self.patients])
        }
        if opt['output_offsets']:
            data['anchor_offsets'] = self.anchor_offsets
            data['agent_offsets'] = self.agent_offsets
            data['patient_offsets'] = self.patient_offsets

        if opt['hidden_eval']:
            data['event_prob'] = self.event_prob
            data['argument_prob'] = self.argument_prob

        if len(self.__en_ver) > 0:
            data['en_ver'] = self.__en_ver

        return data


class SpanSet:
    def __init__(self, *, span_set_name, spans):
        self.__spans = spans
        self.__ss_id = span_set_name

    @property
    def spans(self):
        return self.__spans

    @property
    def ss_id(self):
        return self.__ss_id

    def to_json_dict(self):
        spans = []
        for span in self.spans:
            spans.append({'string': span.ori_text})
        data = {
            'spans': spans,
            'ssid': self.ss_id
        }
        return data


class Span:
    def __init__(self, ori_text):
        self.__text, self.__ori_text, _ = normalize_text(ori_text)

    def __eq__(self, other):
        if isinstance(other, Span):
            return self.__text == other.__text
        return NotImplemented

    @property
    def text(self):  # normalized
        return self.__text

    @property
    def ori_text(self):
        return self.__ori_text


class EmptyCorpus:
    def __init__(self, init_info):
        self.__init_time = datetime.now()
        self.__lang = init_info['lang']
        self.__corpus_id = init_info['corpus-id']
        self.__format_type = init_info['format-type']
        self.__format_version = init_info['format-version']
        self.__provenance = init_info['provenance']
        self.__parsing = False

        assert (self.__format_type == __FORMAT_TYPE__)
        self.__docs = dict()
        self.__sentences = []

    @property
    def lang(self):
        return self.__lang

    @property
    def format_type(self):
        return self.__format_type

    @property
    def format_version(self):
        return self.__format_version

    @property
    def docs(self):
        """
        Returns:
            dict
        """
        return self.__docs

    @property
    def sentences(self):
        return self.__sentences

    @property
    def eid2sent(self):
        return self.__eid2sent

    def clear_annotation(self):
        for doc in self.docs.values():
            doc.clear_annotation()

    def add_sentences(self, sentences, normalize=True):
        for sent in sentences:
            doc_id = sent.doc_id
            if doc_id not in self.__docs:
                self.__docs[doc_id] = Document(doc_id, lang=self.lang, parsing=self.__parsing)
            entry_value = sent.to_json_dict()
            self.__docs[doc_id].add_entry(entry_value, normalize)

        for doc in self.__docs.values():
            doc.sort_sentences()
            self.__sentences += doc.sentences

        self.__eid2sent = dict()
        for sent in self.__sentences:
            self.__eid2sent[sent.entry_id] = sent

    def add_raw_sentences(self, doc_id, sentences):
        for sent in sentences:
            if doc_id not in self.__docs:
                self.__docs[doc_id] = Document(doc_id, lang=self.lang, parsing=self.__parsing)
            entry_value = sent.to_json_dict()
            self.__docs[doc_id].add_entry(entry_value)

        for doc in self.__docs.values():
            doc.sort_sentences()
            self.__sentences += doc.sentences

        self.__eid2sent = dict()
        for sent in self.__sentences:
            self.__eid2sent[sent.entry_id] = sent

    def save(self, output_file):
        entries = {}
        for _, doc in self.docs.items():
            doc_entries = doc.to_json_dict()
            for entry_id, entry_value in doc_entries.items():
                assert (entry_id not in entries)
                entries[entry_id] = entry_value
        data = {
            'corpus-id': self.__corpus_id,
            'entries': entries,
            'format-type': self.__format_type,
            'format-version': self.__format_version,
            'provenance': self.__provenance
        }
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(
                data, output, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == '__main__':
    text = '''وأقرا ما كتبتموه بأسرع وقت … كما قلت لك ياصديقي هذا
عزائي الوحيد … كوني أطلع على منجز أصدقائي هذا يعني لي الكثير
ارجوأن تزودني بكل ما هو جديدك … ماذا عن الاخراج في السينما هل هناك
مشاريع جديدة ؟'''
    normalize_text(text, do_mapping=True)
