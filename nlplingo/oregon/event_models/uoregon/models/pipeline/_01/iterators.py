from .local_constants import *
import random, torch
from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.define_opt import opt
import numpy as np
from collections import defaultdict
from nlplingo.oregon.event_models.uoregon.layers.crf import *
from nlplingo.oregon.event_models.uoregon.tools.xlmr import xlmr_tokenizer

upos_map = {"[PAD]": 0, "[UNK]": 1, "ADP": 2, "DET": 3, "NOUN": 4, "PROPN": 5, "NUM": 6, "PUNCT": 7, "VERB": 8,
            "ADJ": 9, "PRON": 10, "ADV": 11, "AUX": 12, "PART": 13, "CCONJ": 14, "SCONJ": 15, "INTJ": 16, "SYM": 17,
            "X": 18}
xpos_map = {"[PAD]": 0, "[UNK]": 1, "IN": 2, "DT": 3, "NN": 4, "NNP": 5, "CD": 6, ",": 7, "VBD": 8, "NNS": 9, "JJ": 10,
            "PRP": 11, "RB": 12, "VBN": 13, "WDT": 14, "VBG": 15, "TO": 16, "VB": 17, "RP": 18, ".": 19, "CC": 20,
            "EX": 21, "POS": 22, "WP": 23, "PRP$": 24, "HYPH": 25, "WRB": 26, "VBZ": 27, "JJR": 28, "MD": 29, "VBP": 30,
            "''": 31, "``": 32, ":": 33, "NNPS": 34, "JJS": 35, "-LRB-": 36, "-RRB-": 37, "PDT": 38, "UH": 39,
            "RBR": 40, "RBS": 41, "$": 42, "FW": 43, "ADD": 44, "WP$": 45, "SYM": 46, "LS": 47, "NFP": 48, "AFX": 49}
deprel_map = {"[PAD]": 0, "[UNK]": 1, "case": 2, "det": 3, "obl": 4, "nmod": 5, "nummod": 6, "punct": 7, "nsubj": 8,
              "root": 9, "obj": 10, "appos": 11, "conj": 12, "advmod": 13, "acl": 14, "amod": 15, "aux": 16, "mark": 17,
              "xcomp": 18, "compound": 19, "cc": 20, "expl": 21, "ccomp": 22, "parataxis": 23, "flat": 24, "cop": 25,
              "advcl": 26, "csubj": 27, "fixed": 28, "discourse": 29, "iobj": 30, "vocative": 31, "orphan": 32,
              "list": 33, "goeswith": 34}
ner_map = {"[PAD]": 0, "[UNK]": 1, "O": 2,
           "B-DATE": 3, "I-DATE": 4, "E-DATE": 5, "S-DATE": 6,
           "B-CARDINAL": 7, "I-CARDINAL": 1, "E-CARDINAL": 1, "S-CARDINAL": 1,
           "B-PERSON": 1, "I-PERSON": 1, "E-PERSON": 1, "S-PERSON": 1,
           "B-EVENT": 1, "I-EVENT": 1, "E-EVENT": 1, "S-EVENT": 1,
           "B-QUANTITY": 1, "I-QUANTITY": 1, "E-QUANTITY": 1, "S-QUANTITY": 1,
           "B-TIME": 1, "I-TIME": 1, "E-TIME": 1, "S-TIME": 1,
           "B-ORG": 1, "I-ORG": 1, "E-ORG": 1, "S-ORG": 1,
           "B-GPE": 1, "I-GPE": 1, "E-GPE": 1, "S-GPE": 1,
           "B-NORP": 1, "I-NORP": 1, "E-NORP": 1, "S-NORP": 1,
           "B-WORK_OF_ART": 1, "I-WORK_OF_ART": 1, "E-WORK_OF_ART": 1, "S-WORK_OF_ART": 1,
           "B-ORDINAL": 1, "I-ORDINAL": 1, "E-ORDINAL": 1, "S-ORDINAL": 1,
           "B-FAC": 1, "I-FAC": 1, "E-FAC": 1, "S-FAC": 1,
           "B-PRODUCT": 1, "I-PRODUCT": 1, "E-PRODUCT": 1, "S-PRODUCT": 1,
           "B-LOC": 1, "I-LOC": 1, "E-LOC": 1, "S-LOC": 1,
           "B-PERCENT": 1, "I-PERCENT": 1, "E-PERCENT": 1, "S-PERCENT": 1,
           "B-MONEY": 1, "I-MONEY": 1, "E-MONEY": 1, "S-MONEY": 1,
           "B-LANGUAGE": 1, "I-LANGUAGE": 1, "E-LANGUAGE": 1, "S-LANGUAGE": 1,
           "B-LAW": 1, "E-LAW": 1, "I-LAW": 1, "S-LAW": 1}


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


biw2v_map = load_embedding_maps()


def get_bio_tags(agent_extents, patient_extents, word_list):
    """
    example of "agents": [[0, 1]]
    example of "patients": [[20], [28, 29]]
    """
    print('======== iterators.get_bio_tags START =========')
    """
    agent_extents= [[15, 16], [20, 21]]
    patient_extents= [[30, 31]]
    word_list= ['That', 'promise', 'is', 'particularly', 'problematic', 'in', 'the', 'Senate', ',', 'where', 'two', 'pro-abortion', 'rights', 'senators', ',', 'Susan', 'Collins', 'of', 'Maine', 'and', 'Lisa', 'Murkowski', 'of', 'Alaska', ',', 'wo', "n't", 'commit', 'to', 'approving', 'the', 'bill', 'with', 'the', 'Planned', 'Parenthood', 'provision', 'in', 'it', '.']
    entity_tags= [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 3, 3, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 6, 7, 3, 3, 3, 3, 3, 3, 3, 3]
    """
    print('agent_extents=', agent_extents)
    print('patient_extents=', patient_extents)
    print('word_list=', word_list)
    agent_labels = ['O' for _ in range(len(word_list))]
    patient_labels = ['O' for _ in range(len(word_list))]
    for agent_extent in agent_extents:
        for k, agent_tok in enumerate(agent_extent):
            if k == 0:
                agent_labels[agent_tok] = 'B-AGENT'
            else:
                agent_labels[agent_tok] = 'I-AGENT'

    for patient_extent in patient_extents:
        for k, patient_tok in enumerate(patient_extent):
            if k == 0:
                patient_labels[patient_tok] = 'B-PATIENT'
            else:
                patient_labels[patient_tok] = 'I-PATIENT'
    entity_tags = []
    for k in range(len(word_list)):
        if agent_labels[k] == patient_labels[k]:
            tag = 'O'
        elif 'O' in [agent_labels[k], patient_labels[k]]:
            if agent_labels[k] != 'O':
                tag = agent_labels[k]
            else:
                tag = patient_labels[k]
        else:
            tag = '{}|{}'.format(agent_labels[k], patient_labels[k])
        entity_tags.append(ARGUMENT_TAG_MAP[tag])   # These tags are not used in this function: "<PAD>": 0, "<SOS>": 1, "<EOS>"
    # entity_tags = [ARGUMENT_TAG_MAP[SOS]] + entity_tags
    print('entity_tags=', entity_tags)
    print('======== iterators.get_bio_tags END =========')
    return entity_tags


def get_arguments(id2tag, tag_ids, ori_example, seperate_outputs=True):
    print('============ iterators.get_arguments START ==========')
    """
    id2tag= {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: 'O', 4: 'B-AGENT', 5: 'I-AGENT', 6: 'B-PATIENT', 7: 'I-PATIENT', 8: 'B-AGENT|B-PATIENT', 9: 'B-AGENT|I-PATIENT', 10: 'I-AGENT|B-PATIENT', 11: 'I-AGENT|I-PATIENT'}
    tag_ids= tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 6, 7, 3, 3, 3, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
    ori_text=
       ﻕﺎﻟ ﻢﺘﺣﺪﺛ ﺏﺎﺴﻣ ﺎﻠﻣﺪﻌﻳ ﺎﻠﻋﺎﻣ ﺍﻸﻣﺮﻴﻜﻳ ﺝﻭﺰﻴﻓ ﺪﻴﺠﻴﻧﻮﻓﺍ ﺈﻧ ﺪﻴﺠﻴﻧﻮﻓﺍ ﻞﻣ ﻲﻜﻧ ﻢﺗﺄﻛﺩﺍ ﻢﻣﺍ ﺇﺫﺍ ﻙﺎﻧ ﺲﻴﺣﺎﻜﻣ ﺎﻠﻤﺘﻬﻤﻴﻧ ﺎﻠﺨﻤﺳﺓ ﺄﻣ ﺲﻴﺴﻘﻃ ﺎﻠﺘﻬﻣ.
    actual_length= 23
    tag_ids= [3 3 3 3 3 3 3 3 3 4 3 3 3 3 3 3 3 6 7 3 3 3 3]
    tags= ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-AGENT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PATIENT', 'I-PATIENT', 'O', 'O', 'O', 'O']
    agent_tags= ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-AGENT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    patient_tags= ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PATIENT', 'I-PATIENT', 'O', 'O', 'O', 'O']
    ori_example[span]= [[4, 6], [8, 12], [14, 17], [19, 24], [26, 30], [32, 39], [41, 45], [47, 54], [56, 57], [59, 66], [68, 69], [71, 73], [75, 80], [82, 84], [86, 88], [90, 92], [94, 99], [101, 108], [110, 115], [117, 118], [120, 124], [126, 130], [131, 131]]
    A: start_span= [59, 66] end_span= [59, 66] text_span= ﺪﻴﺠﻴﻧﻮﻓﺍ
    B: start_span= [101, 108] end_span= [110, 115] text_span= ﺎﻠﻤﺘﻬﻤﻴﻧ ﺎﻠﺨﻤﺳﺓ
    """
    print('id2tag=', id2tag)
    print('tag_ids=', tag_ids)

    ori_text = ori_example['text']
    actual_length = len(ori_example['word'])
    print('ori_text=', ori_text)
    print('actual_length=', actual_length)

    tag_ids = tag_ids.long().data.cpu().numpy()
    tag_ids = tag_ids[: actual_length]
    print('tag_ids=', tag_ids)
    tags = [id2tag[tag_id] for tag_id in tag_ids]
    print('tags=', tags)
    agent_tags = []
    patient_tags = []
    for tag in tags:
        if 'AGENT' not in tag and 'PATIENT' not in tag:
            agent_tags.append('O')
            patient_tags.append('O')
        else:
            sub_tags = tag.split('|')
            if len(sub_tags) == 1:
                if 'AGENT' in sub_tags[0]:
                    agent_tags.append(sub_tags[0])
                    patient_tags.append('O')
                else:
                    agent_tags.append('O')
                    patient_tags.append(sub_tags[0])
            else:
                agent_tags.append(sub_tags[0])
                patient_tags.append(sub_tags[1])
    print('agent_tags=', agent_tags)
    print('patient_tags=', patient_tags)
    assert len(agent_tags) == len(patient_tags)
    agents = []
    patients = []

    agent_offsets = {}
    patient_offsets = {}

    print('ori_example[span]=', ori_example['span'])
    current_agent = []
    current_patient = []
    for k in range(actual_length):
        if len(current_agent) > 0 and (agent_tags[k] == 'O' or agent_tags[k].startswith('B-')):
            start_span = ori_example['span'][current_agent[0]]
            end_span = ori_example['span'][current_agent[-1]]

            text_span = ori_text[start_span[0]: end_span[1] + 1]
            print('A: start_span=', start_span, 'end_span=', end_span, 'text_span=', text_span)
            agents.append(text_span)
            agent_offsets[text_span] = [start_span[0], end_span[1] + 1]
            current_agent = []

        if 'AGENT' in agent_tags[k]:
            current_agent.append(k)

        if len(current_patient) > 0 and (patient_tags[k] == 'O' or patient_tags[k].startswith('B-')):
            start_span = ori_example['span'][current_patient[0]]
            end_span = ori_example['span'][current_patient[-1]]

            text_span = ori_text[start_span[0]: end_span[1] + 1]
            print('B: start_span=', start_span, 'end_span=', end_span, 'text_span=', text_span)
            patients.append(text_span)
            patient_offsets[text_span] = [start_span[0], end_span[1] + 1]
            current_patient = []

        if 'PATIENT' in patient_tags[k]:
            current_patient.append(k)

    if len(current_agent) > 0:
        start_span = ori_example['span'][current_agent[0]]
        end_span = ori_example['span'][current_agent[-1]]

        text_span = ori_text[start_span[0]: end_span[1] + 1]
        print('C: start_span=', start_span, 'end_span=', end_span, 'text_span=', text_span)
        agents.append(text_span)
        agent_offsets[text_span] = [start_span[0], end_span[1] + 1]

    if len(current_patient) > 0:
        start_span = ori_example['span'][current_patient[0]]
        end_span = ori_example['span'][current_patient[-1]]

        text_span = ori_text[start_span[0]: end_span[1] + 1]
        print('D: start_span=', start_span, 'end_span=', end_span, 'text_span=', text_span)
        patients.append(text_span)
        patient_offsets[text_span] = [start_span[0], end_span[1] + 1]

    print('============ iterators.get_arguments START ==========')

    if seperate_outputs:
        return agents, patients, agent_offsets, patient_offsets
    else:
        return agents + patients


class EDIterator:
    def __init__(self, xlmr_model, data_path, is_eval_data=False):
        print('Loading {} for ED iterator...'.format(data_path))
        self.opt = opt
        self.xlmr_model = xlmr_model
        self.xlmr_model.eval()
        self.is_eval_data = is_eval_data
        self.data_path = data_path
        self.encoded_data = self.encode_data()
        self.num_examples = len(self.encoded_data['english'] + self.encoded_data['arabic'])
        self.data_batches = self.create_batches()
        self.num_batches = len(self.data_batches)

    def get_ED_labels(self, word_list, trigger_list, event_types):
        '''
        Each event has only 1 trigger.
        Each trigger can be composed of multiple words.
        '''
        ED_labels = [0 for _ in range(len(word_list))]
        for event_id in range(len(trigger_list)):
            trigger_ids = trigger_list[event_id]
            ED_labels[trigger_ids[0]] = EVENT_MAP[event_types[event_id]]    # YS: WARNING this is not BIO. This is just a single token!!
        return ED_labels

    def encode_example(self, example):
        word_list = example['word']
        upos_list = example['upos']
        xpos_list = example['xpos']
        head_list = example['head']
        dep_rel_list = example['dep_rel']
        ner_list = example['ner']
        lang_weight = 1.0 if opt['co_train_lambda'] == 0 or example['lang'] == 'english' else opt['co_train_lambda']
        # *****************************
        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list)
        # ****** biw2v ************
        biw2v_ids = [biw2v_map.get(word.lower(), biw2v_map[UNK_TOKEN]) for word in word_list]
        # *****************************
        upos_ids = [upos_map.get(upos, upos_map[UNK_TOKEN]) for upos in upos_list]
        xpos_ids = [xpos_map.get(xpos, xpos_map[UNK_TOKEN]) for xpos in xpos_list]
        head_ids = head_list
        dep_rel_ids = [deprel_map.get(dep_rel, deprel_map[UNK_TOKEN]) for dep_rel in dep_rel_list]
        ner_ids = [ner_map.get(ner, ner_map[UNK_TOKEN]) for ner in ner_list]

        ED_labels = self.get_ED_labels(word_list, example['triggers'], example['event-types'])

        """ Example:
        #### Before encoding
        'word': ['On', 'the', 'afternoon', 'of', 'Oct.', '7', ',', '1974', ',', 'a', 'mob', 'of', '200', 'enraged', 'whites', ',', 'many', 'of', 'them', 'students', ',', 'closed', 'in', 'on', 'a', 'bus', 'filled', 'with', 'black', 'students', 'that', 'was', 'trying', 'to', 'pull', 'away', 'from', 'the', 'local', 'high', 'school', '.']
        'lemma': ['on', 'the', 'afternoon', 'of', 'Oct.', '7', ',', '1974', ',', 'a', 'mob', 'of', '200', 'enrage', 'white', ',', 'many', 'of', 'they', 'student', ',', 'close', 'in', 'on', 'a', 'bus', 'fill', 'with', 'black', 'student', 'that', 'be', 'try', 'to', 'pull', 'away', 'from', 'the', 'local', 'high', 'school', '.']
        'upos': ['ADP', 'DET', 'NOUN', 'ADP', 'PROPN', 'NUM', 'PUNCT', 'NUM', 'PUNCT', 'DET', 'NOUN', 'ADP', 'NUM', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'ADP', 'PRON', 'NOUN', 'PUNCT', 'VERB', 'ADV', 'ADP', 'DET', 'NOUN', 'VERB', 'ADP', 'ADJ', 'NOUN', 'PRON', 'AUX', 'VERB', 'PART', 'VERB', 'ADP', 'ADP', 'DET', 'ADJ', 'ADJ', 'NOUN', 'PUNCT']
        'xpos': ['IN', 'DT', 'NN', 'IN', 'NNP', 'CD', ',', 'CD', ',', 'DT', 'NN', 'IN', 'CD', 'VBD', 'NNS', ',', 'JJ', 'IN', 'PRP', 'NNS', ',', 'VBD', 'RB', 'IN', 'DT', 'NN', 'VBN', 'IN', 'JJ', 'NNS', 'WDT', 'VBD', 'VBG', 'TO', 'VB', 'RP', 'IN', 'DT', 'JJ', 'JJ', 'NN', '.']
        'morph': ['_', 'Definite=Def|PronType=Art', 'Number=Sing', '_', 'Number=Sing', 'NumType=Card', '_', 'NumType=Card', '_', 'Definite=Ind|PronType=Art', 'Number=Sing', '_', 'NumType=Card', 'Tense=Past|VerbForm=Fin', 'Number=Plur', '_', 'Degree=Pos', '_', 'Case=Acc|Number=Plur|Person=3|PronType=Prs', 'Number=Plur', '_', 'Mood=Ind|Tense=Past|VerbForm=Fin', '_', '_', 'Definite=Ind|PronType=Art', 'Number=Sing', 'Tense=Past|VerbForm=Part', '_', 'Degree=Pos', 'Number=Plur', 'PronType=Rel', 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', 'Tense=Pres|VerbForm=Part', '_', 'VerbForm=Inf', '_', '_', 'Definite=Def|PronType=Art', 'Degree=Pos', 'Degree=Pos', 'Number=Sing', '_']
        'head': [3, 3, 14, 5, 3, 5, 5, 5, 14, 11, 14, 13, 11, 0, 14, 15, 15, 19, 17, 17, 22, 14, 22, 26, 26, 22, 26, 30, 30, 27, 33, 33, 30, 35, 33, 35, 41, 41, 41, 41, 35, 14]
        'dep_rel': ['case', 'det', 'obl', 'case', 'nmod', 'nummod', 'punct', 'nummod', 'punct', 'det', 'nsubj', 'case', 'nmod', 'root', 'obj', 'punct', 'appos', 'case', 'nmod', 'nmod', 'punct', 'conj', 'advmod', 'case', 'det', 'obl', 'acl', 'case', 'amod', 'obl', 'nsubj', 'aux', 'acl', 'mark', 'xcomp', 'compound', 'case', 'det', 'amod', 'amod', 'obl', 'punct']
        'ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        'span': [[0, 1], [3, 5], [7, 15], [17, 18], [20, 23], [25, 25], [26, 26], [28, 31], [32, 32], [34, 34], [36, 38], [40, 41], [43, 45], [47, 53], [55, 60], [61, 61], [63, 66], [68, 69], [71, 74], [76, 83], [84, 84], [86, 91], [93, 94], [96, 97], [99, 99], [101, 103], [105, 110], [112, 115], [117, 121], [123, 130], [132, 135], [137, 139], [141, 146], [148, 149], [151, 154], [156, 159], [161, 164], [166, 168], [170, 174], [176, 179], [181, 186], [187, 187]]
        
        #### After encoding
        xlmr_ids:
        tensor([     0,   2161,     70, 157109,    111,  33649,      5,    361,      6,
                     4,  27898,      6,      4,     10,  81158,    111,   1781,     22,
                 29838,     71,  35011,      7,      6,      4,   5941,    111,   2856,
                 25921,      6,      4, 155738,     23,     98,     10,   5324, 152382,
                   678,  22556,  25921,    450,    509,  31577,     47,  50065,  16065,
                  1295,     70,   4000,  11192,  10696,      6,      5,      2])

        biw2v_ids: len=42,      [11, 3, 2078, 5, 2331, 244, 4, 3939, 4, 10, 12556, 5, 1962, 25452, 17003, 4, 224, 5, 148, 1638, 4, 1393, 9, 11, 10, 2610, 5094, 19, 3225, 1638, 14, 24, 1931, 8, 5169, 1828, 30, 3, 245, 210, 1096, 6]
        retrieve_ids, len=42,   [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15, 16, 17, 20, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51]
        upos_ids, len=42,       [2, 3, 4, 2, 5, 6, 7, 6, 7, 3, 4, 2, 6, 8, 4, 7, 9, 2, 10, 4, 7, 8, 11, 2, 3, 4, 8, 2, 9, 4, 10, 12, 8, 13, 8, 2, 2, 3, 9, 9, 4, 7]
        xpos_ids, len=42,       [2, 3, 4, 2, 5, 6, 7, 6, 7, 3, 4, 2, 6, 8, 9, 7, 10, 2, 11, 9, 7, 8, 12, 2, 3, 4, 13, 2, 10, 9, 14, 8, 15, 16, 17, 18, 2, 3, 10, 10, 4, 19]
        head_ids, len=42,       [3, 3, 14, 5, 3, 5, 5, 5, 14, 11, 14, 13, 11, 0, 14, 15, 15, 19, 17, 17, 22, 14, 22, 26, 26, 22, 26, 30, 30, 27, 33, 33, 30, 35, 33, 35, 41, 41, 41, 41, 35, 14]
        dep_rel_ids, len=42,    [2, 3, 4, 2, 5, 6, 7, 6, 7, 3, 8, 2, 5, 9, 10, 7, 11, 2, 5, 5, 7, 12, 13, 2, 3, 4, 14, 2, 15, 4, 8, 16, 14, 17, 18, 19, 2, 3, 15, 15, 4, 7]
        ner_ids, len=42,        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        lang_weight             1.0
        ED_labels, len=42,      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0]
        
        """

        return xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, lang_weight, ED_labels

    # for each SequenceExample, generate features to become encoded_ex
    # shuffle encoded_ex
    def encode_data(self):
        data = read_json(self.data_path)
        encoded_data = {
            'english': [],
            'arabic': []
        }
        for data_point in data:
            encoded_ex = self.encode_example(data_point)
            # ********* skip over-length examples ******************************
            xlmr_ids = encoded_ex[0]
            if xlmr_ids is not None and len(xlmr_ids) <= SAFE_BERT_TOKENS:
                if data_point['lang'] == 'english':
                    encoded_data['english'].append(encoded_ex)
                else:
                    encoded_data['arabic'].append(encoded_ex)
        # shuffle for training
        if not self.is_eval_data:
            encoded_data['english'] = shuffle_list(encoded_data['english'])
            encoded_data['arabic'] = shuffle_list(encoded_data['arabic'])
        return encoded_data

    def create_batches(self):
        en_ratio = 1. * len(self.encoded_data['english']) / (
                len(self.encoded_data['arabic']) + len(self.encoded_data['english']))
        if en_ratio == 1.0:
            batches = [self.encoded_data['english'][i:i + opt['batch_size']] for i in
                       range(0, self.num_examples, opt['batch_size'])]
            return batches
        elif en_ratio == 0:
            batches = [self.encoded_data['arabic'][i:i + opt['batch_size']] for i in
                       range(0, self.num_examples, opt['batch_size'])]
            return batches
        else:
            en_size = int(opt['batch_size'] * en_ratio)
            ar_size = opt['batch_size'] - en_size
            en_batches = [self.encoded_data['english'][i:i + en_size] for i in
                          range(0, len(self.encoded_data['english']), en_size)]
            ar_batches = [self.encoded_data['arabic'][i:i + ar_size] for i in
                          range(0, len(self.encoded_data['arabic']), ar_size)] if ar_size > 0 else []
            batches = []
            for bid in range(len(en_batches)):
                if bid < len(ar_batches):
                    batch = en_batches[bid] + ar_batches[bid]
                    batch = shuffle_list(batch)
                else:
                    batch = en_batches[bid]
                batches.append(batch)
            return batches

    def __len__(self):
        return len(self.data_batches)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data_batches):
            raise IndexError
        batch = self.data_batches[key]
        batch_size = len(batch)

        # E.g. if: l = [('1a','1b'),('2a','2b'),('3a','3b'),('4a','4b')]
        # zip(*l): [('1a', '2a', '3a', '4a'), ('1b', '2b', '3b', '4b')]
        batch = list(zip(*batch))

        lens = [len(x) for x in batch[0]]   # batch[0] is list[xlmr_ids], so this gives: [len(xlmr_ids) for each example]
        batch, _ = sort_all(batch, lens)    # sort elements in batch by decreasing order of their len
        # convert to tensors
        xlmr_ids = do_padding(batch[0], batch_size)
        biw2v_ids = do_padding(batch[1], batch_size)
        # ***********************************************
        retrieve_ids = do_padding(batch[2], batch_size)
        upos_ids = do_padding(batch[3], batch_size)
        xpos_ids = do_padding(batch[4], batch_size)

        head_ids = do_padding(batch[5], batch_size)
        deprel_ids = do_padding(batch[6], batch_size)
        ner_ids = do_padding(batch[7], batch_size)
        lang_weights = torch.Tensor(batch[8])

        ED_labels = do_padding(batch[9], batch_size)

        # If:   retrieve_ids = torch.LongTensor([1, 2, 3, 0, 5])
        # Then: torch.eq(retrieve_ids, 0) produces: tensor([False, False, False,  True, False])
        pad_masks = torch.eq(retrieve_ids, 0)

        return (
            xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights,
            ED_labels,
            pad_masks)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def shuffle_batches(self):
        indices = list(range(len(self.data_batches)))
        random.shuffle(indices)
        self.data_batches = [self.data_batches[i] for i in indices]


class ArgumentIterator:
    def __init__(self, xlmr_model, data_path, is_eval_data=False):
        print('Using {} for entity iterator...'.format(data_path))
        self.opt = opt
        self.xlmr_model = xlmr_model
        self.xlmr_model.eval()
        self.is_eval_data = is_eval_data
        self.data_path = data_path

        self.id2ori_example = {}
        self.id2tag = dict([(v, k) for k, v in ARGUMENT_TAG_MAP.items()])

        self.encoded_data = self.encode_data()
        self.num_examples = len(self.encoded_data['english'] + self.encoded_data['arabic'])

        self.data_batches = self.create_batches()
        self.num_batches = len(self.data_batches)

    def encode_example(self, example):
        """
        example of "span": [[0, 7], [9, 17], [19, 20], [22, 27], ...
        There are as many sublists as there are tokens. Each sublist refers to the [starting-char, ending-char] of each token

        "text": raw text string of the sentence

        example of "trigger": [13]
        example of "agents": [[0, 1]]
        example of "patients": [[20], [28, 29]]

        ###############

        example['word'] ['A', 'number', 'of', 'National', 'Football', 'League', '(', 'NFL', ')', 'players', 'have', 'protested', 'after', 'being', 'attacked', 'by', 'the', 'US', 'president', '.']
        example['upos'] ['DET', 'NOUN', 'ADP', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'NOUN', 'AUX', 'VERB', 'SCONJ', 'AUX', 'VERB', 'ADP', 'DET', 'PROPN', 'PROPN', 'PUNCT']
        example['xpos'] ['DT', 'NN', 'IN', 'NNP', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', 'NNS', 'VBP', 'VBN', 'IN', 'VBG', 'VBN', 'IN', 'DT', 'NNP', 'NNP', '.']
        example['head'] [2, 12, 6, 6, 6, 2, 8, 6, 8, 12, 12, 0, 15, 15, 12, 19, 19, 19, 15, 12]
        example['dep_rel'] ['det', 'nsubj', 'case', 'compound', 'compound', 'nmod', 'punct', 'appos', 'punct', 'nsubj', 'aux', 'root', 'mark', 'aux', 'advcl', 'case', 'det', 'compound', 'obl', 'punct']
        example['ner'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        lang_weight 1.0
        trigger_toks [11]
        trigger_word protested

        ###############

        xlmr_ids    tensor([     0,     62,  14012,    111,   9907,  98809,  19175,     15, 186831,
                              1388,  92865,    765,  18782,    297,   7103,   8035,  52875,    297,
                               390,     70,   7082,  13918,      6,      5,      2,      2,  18782,
                               297,      2])
        NOTE: the trailing [2,  18782, 297, 2] comes from the trigger word "protested"

        biw2v_ids       [10, 166, 5, 76, 5067, 1764, 16, 105158, 17, 5313, 40, 7923, 125, 200, 2781, 18, 3, 99, 84, 6]
        retrieve_ids    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23]
        upos_ids        [3, 4, 2, 5, 5, 5, 7, 5, 7, 4, 12, 8, 15, 12, 8, 2, 3, 5, 5, 7]
        xpos_ids        [3, 4, 2, 5, 5, 5, 36, 5, 37, 9, 30, 13, 2, 15, 13, 2, 3, 5, 5, 19]
        head_ids        [2, 12, 6, 6, 6, 2, 8, 6, 8, 12, 12, 0, 15, 15, 12, 19, 19, 19, 15, 12]
        dep_rel_ids     [3, 8, 2, 19, 19, 5, 7, 11, 7, 8, 16, 9, 17, 16, 26, 2, 3, 19, 4, 7]
        ner_ids         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        lang_weight     1.0
        trigger_tok     11
        entity_tags     [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3]
        eid             0
        """
        word_list = example['word']
        upos_list = example['upos']
        xpos_list = example['xpos']
        head_list = example['head']
        dep_rel_list = example['dep_rel']
        ner_list = example['ner']
        lang_weight = 1.0 if opt['co_train_lambda'] == 0 or example['lang'] == 'english' else opt['co_train_lambda']
        # *****************************
        trigger_toks = example['trigger']

        # get raw text string of the trigger
        trigger_word = example['text'][
                       example['span'][int(trigger_toks[0])][0]: example['span'][int(trigger_toks[-1])][1] + 1]

        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list, trigger_word)
        # ****** biw2v ************
        biw2v_ids = [biw2v_map.get(word.lower(), biw2v_map[UNK_TOKEN]) for word in word_list]
        # *****************************
        upos_ids = [upos_map.get(upos, upos_map[UNK_TOKEN]) for upos in upos_list]
        xpos_ids = [xpos_map.get(xpos, xpos_map[UNK_TOKEN]) for xpos in xpos_list]
        head_ids = head_list
        dep_rel_ids = [deprel_map.get(dep_rel, deprel_map[UNK_TOKEN]) for dep_rel in dep_rel_list]
        ner_ids = [ner_map.get(ner, ner_map[UNK_TOKEN]) for ner in ner_list]

        entity_tags = get_bio_tags(
            agent_extents=example['agents'],
            patient_extents=example['patients'],
            word_list=word_list
        )
        trigger_tok = example['trigger'][0]
        eid = len(self.id2ori_example)
        self.id2ori_example[eid] = example
        return xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, lang_weight, trigger_tok, entity_tags, eid

    def encode_data(self):
        data = read_json(self.data_path)
        encoded_data = {
            'english': [],
            'arabic': []
        }
        for data_point in data:
            encoded_ex = self.encode_example(data_point)
            # ********* remove over-length examples ******************************
            xlmr_ids = encoded_ex[0]
            if xlmr_ids is not None and len(xlmr_ids) <= SAFE_BERT_TOKENS:  # for safe run
                if data_point['lang'] == 'english':
                    encoded_data['english'].append(encoded_ex)
                else:
                    encoded_data['arabic'].append(encoded_ex)
        # shuffle for training
        if not self.is_eval_data:
            encoded_data['english'] = shuffle_list(encoded_data['english'])
            encoded_data['arabic'] = shuffle_list(encoded_data['arabic'])
        return encoded_data

    def create_batches(self):
        en_ratio = 1. * len(self.encoded_data['english']) / (
                len(self.encoded_data['arabic']) + len(self.encoded_data['english']))
        if en_ratio == 1.0:
            batches = [self.encoded_data['english'][i:i + opt['batch_size']] for i in
                       range(0, self.num_examples, opt['batch_size'])]
            return batches
        elif en_ratio == 0:
            batches = [self.encoded_data['arabic'][i:i + opt['batch_size']] for i in
                       range(0, self.num_examples, opt['batch_size'])]
            return batches
        else:
            en_size = int(opt['batch_size'] * en_ratio)
            ar_size = opt['batch_size'] - en_size
            en_batches = [self.encoded_data['english'][i:i + en_size] for i in
                          range(0, len(self.encoded_data['english']), en_size)]
            ar_batches = [self.encoded_data['arabic'][i:i + ar_size] for i in
                          range(0, len(self.encoded_data['arabic']), ar_size)] if ar_size > 0 else []
            batches = []
            for bid in range(len(en_batches)):
                if bid < len(ar_batches):
                    batch = en_batches[bid] + ar_batches[bid]
                    batch = shuffle_list(batch)
                else:
                    batch = en_batches[bid]
                batches.append(batch)
            return batches

    def __len__(self):
        return len(self.data_batches)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data_batches):
            raise IndexError
        batch = self.data_batches[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        lens = [len(x) for x in batch[0]]
        batch, _ = sort_all(batch, lens)
        # xlmr_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, trigger_tok, entity_tags
        # convert to tensors
        xlmr_ids = do_padding(batch[0], batch_size)
        biw2v_ids = do_padding(batch[1], batch_size)
        # **************************************************
        retrieve_ids = do_padding(batch[2], batch_size)
        upos_ids = do_padding(batch[3], batch_size)
        xpos_ids = do_padding(batch[4], batch_size)

        head_ids = do_padding(batch[5], batch_size)
        deprel_ids = do_padding(batch[6], batch_size)
        ner_ids = do_padding(batch[7], batch_size)
        lang_weights = torch.Tensor(batch[8])

        triggers = torch.Tensor(batch[9])
        entity_tags = do_padding(batch[10], batch_size)
        eid = torch.Tensor(batch[11])
        pad_masks = torch.eq(retrieve_ids, 0)
        return (
            xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights,
            triggers, entity_tags,
            eid, pad_masks
        )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def shuffle_batches(self):
        indices = list(range(len(self.data_batches)))
        random.shuffle(indices)
        self.data_batches = [self.data_batches[i] for i in indices]


class PipelineIterator:
    """
    eval -> PipelineTrainer -> PipelineIterator
    """
    def __init__(self, xlmr_model, data_path):
        # e.g. data_path = datapoints/data/arabic-abstract-sample/arabic-abstract-sample.pipeline.test (this is already parsed by stanfordnlp)
        print('Using {} for pipeline iterator...'.format(data_path))

        self.opt = opt
        self.xlmr_model = xlmr_model    # This is the XLMRModel.from_pretrained
        self.xlmr_model.eval()
        self.data_path = data_path

        self.id2ori_example = {}

        self.encoded_data = self.encode_data()
        self.num_examples = len(self.encoded_data)
        print('PipelineIterator: num_examples=', self.num_examples)
        self.data_batches = self.create_batches()
        self.num_batches = len(self.data_batches)
        print('PipelineIterator: num_batches=', self.num_batches)

    def encode_example(self, example):
        word_list = example['word']
        upos_list = example['upos']
        xpos_list = example['xpos']
        head_list = example['head']
        dep_rel_list = example['dep_rel']
        ner_list = example['ner']
        example['norm2ori_offsetmap'] = {int(k): v for k, v in example['norm2ori_offsetmap'].items()}

        # print('len(word_list)=', len(word_list))          20
        # print('len(upos_list)=', len(upos_list))          20
        # print('len(xpos_list)=', len(xpos_list))          20
        # print('len(head_list)=', len(head_list))          20
        # print('len(dep_rel_list)=', len(dep_rel_list))    20
        # print('len(ner_list)=', len(ner_list))            20
        #
        # print("example['norm2ori_offsetmap']=", example['norm2ori_offsetmap'])
        # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
        # 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28,
        # 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41,
        # 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54,
        # 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67,
        # 68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80,
        # 81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90, 91: 91, 92: 92, 93: 93,
        # 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99, 100: 100, 101: 101, 102: 102, 103: 103, 104: 104, 105: 105, 106: 106}
        # *****************************
        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list)
        # print('xlmr_ids.shape=', xlmr_ids.shape)          [29]
        # print('len(retrieve_ids)=', len(retrieve_ids))    20
        #
        # print('retrieve_ids=', retrieve_ids)
        # [1, 2, 4, 6, 7, 8, 9, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]
        #
        # ****** biw2v ************
        biw2v_ids = [biw2v_map.get(word.lower(), biw2v_map[UNK_TOKEN]) for word in word_list]
        # print('len(biw2v_ids)=', len(biw2v_ids))          20
        #
        # print('biw2v_ids=', biw2v_ids)
        # [113306, 116408, 135535, 1, 1, 1, 137810, 176957, 129403, 123127, 177487, 1, 113249, 113236, 1, 113558, 1, 113199, 113472, 6]
        #
        # *****************************
        upos_ids = [upos_map.get(upos, upos_map[UNK_TOKEN]) for upos in upos_list]
        xpos_ids = [xpos_map.get(xpos, xpos_map[UNK_TOKEN]) for xpos in xpos_list]
        head_ids = head_list
        dep_rel_ids = [deprel_map.get(dep_rel, deprel_map[UNK_TOKEN]) for dep_rel in dep_rel_list]
        ner_ids = [ner_map.get(ner, ner_map[UNK_TOKEN]) for ner in ner_list]
        # print('len(upos_ids)=', len(upos_ids))        20
        # print('len(xpos_ids)=', len(xpos_ids))        20
        # print('len(head_ids)=', len(head_ids))        20
        # print('len(dep_rel_ids)=', len(dep_rel_ids))  20
        # print('len(ner_ids)=', len(ner_ids))          20
        eid = len(self.id2ori_example)
        # print('eid=', eid)                            0
        self.id2ori_example[eid] = example

        return xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, eid

    def encode_data(self):
        if self.opt['input_lang'] != 'english' and self.opt['use_dep2sent']:
            data = read_pickle(self.data_path)  # alternative data permutated by dep2sent model
        else:
            data = read_json(self.data_path)

        encoded_data = []
        for data_point in data:
            encoded_ex = self.encode_example(data_point)
            # ********* skip over-length examples ******************************
            xlmr_ids = encoded_ex[0]
            if xlmr_ids is not None and len(xlmr_ids) <= SAFE_BERT_TOKENS:
                encoded_data.append(encoded_ex)
        return encoded_data

    def create_batches(self):
        batches = [self.encoded_data[i:i + opt['batch_size']] for i in range(0, self.num_examples, opt['batch_size'])]
        return batches

    def __len__(self):
        return len(self.data_batches)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data_batches):
            raise IndexError
        batch = self.data_batches[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        lens = [len(x) for x in batch[0]]
        batch, _ = sort_all(batch, lens)
        # xlmr_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, eid
        # convert to tensors
        xlmr_ids = do_padding(batch[0], batch_size)
        biw2v_ids = do_padding(batch[1], batch_size)
        # **********************************************
        retrieve_ids = do_padding(batch[2], batch_size)
        upos_ids = do_padding(batch[3], batch_size)
        xpos_ids = do_padding(batch[4], batch_size)

        head_ids = do_padding(batch[5], batch_size)
        deprel_ids = do_padding(batch[6], batch_size)
        ner_ids = do_padding(batch[7], batch_size)

        eid = torch.Tensor(batch[8])
        pad_masks = torch.eq(retrieve_ids, 0)
        return (
            xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, eid, pad_masks
        )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def shuffle_batches(self):
        indices = list(range(len(self.data_batches)))
        random.shuffle(indices)
        self.data_batches = [self.data_batches[i] for i in indices]


