#from nlplingo.tasks.sequence.run import find_token_indices_of_markers, remove_tokens_at_indices
from nlplingo.tasks.sequence.utils import find_token_indices_of_markers, remove_tokens_at_indices

from nlplingo.oregon.event_models.uoregon.tools.xlmr import xlmr_tokenizer
from nlplingo.oregon.event_models.uoregon.tools.utils import *


upos_map = {'[PAD]': 0, '[UNK]': 1, 'ADJ': 2, 'ADP': 3, 'ADV': 4, 'AUX': 5, 'CCONJ': 6, 'DET': 7, 'INTJ': 8, 'NOUN': 9,
            'NUM': 10, 'PART': 11, 'PRON': 12, 'PROPN': 13, 'PUNCT': 14, 'SCONJ': 15, 'SYM': 16, 'VERB': 17, 'X': 18}

#upos_map = {"[PAD]": 0, "[UNK]": 1, "ADP": 2, "DET": 3, "NOUN": 4, "PROPN": 5, "NUM": 6, "PUNCT": 7, "VERB": 8,
#            "ADJ": 9, "PRON": 10, "ADV": 11, "AUX": 12, "PART": 13, "CCONJ": 14, "SCONJ": 15, "INTJ": 16, "SYM": 17,
#            "X": 18}

# xpos_map = {"[PAD]": 0, "[UNK]": 1, "IN": 2, "DT": 3, "NN": 4, "NNP": 5, "CD": 6, ",": 7, "VBD": 8, "NNS": 9, "JJ": 10,
#             "PRP": 11, "RB": 12, "VBN": 13, "WDT": 14, "VBG": 15, "TO": 16, "VB": 17, "RP": 18, ".": 19, "CC": 20,
#             "EX": 21, "POS": 22, "WP": 23, "PRP$": 24, "HYPH": 25, "WRB": 26, "VBZ": 27, "JJR": 28, "MD": 29, "VBP": 30,
#             "''": 31, "``": 32, ":": 33, "NNPS": 34, "JJS": 35, "-LRB-": 36, "-RRB-": 37, "PDT": 38, "UH": 39,
#             "RBR": 40, "RBS": 41, "$": 42, "FW": 43, "ADD": 44, "WP$": 45, "SYM": 46, "LS": 47, "NFP": 48, "AFX": 49}

# following xpos based on stanfordnlp annotations on SerifXML pos_sequence
xpos_map = {'[PAD]': 0, '[UNK]': 1, '``': 2, ',': 3, ':': 4, '.': 5, "''": 6, '$': 7, 'ADD': 8, 'AFX': 9, 'CC': 10,
            'CD': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'HYPH': 15, 'IN': 16, 'JJ': 17, 'JJR': 18, 'JJS': 19, '-LRB-': 20,
            'LS': 21, 'MD': 22, 'NFP': 23, 'NN': 24, 'NNP': 25, 'NNPS': 26, 'NNS': 27, 'PDT': 28, 'POS': 29, 'PRP': 30,
            'PRP$': 31, 'RB': 32, 'RBR': 33, 'RBS': 34, 'RP': 35, '-RRB-': 36, 'SYM': 37, 'TO': 38, 'UH': 39, 'VB': 40,
            'VBD': 41, 'VBG': 42, 'VBN': 43, 'VBP': 44, 'VBZ': 45, 'WDT': 46, 'WP': 47, 'WP$': 48, 'WRB': 49}

# following xpos based on original SerifXML pos-tags from parse tree
# xpos_map = {'[PAD]': 0, '[UNK]': 1, '``': 2, ',': 3, ':': 4, '.': 5, "''": 6, '$': 7, 'CC': 8, 'CD': 9, 'DATE-NNP': 10,
#             'DATE-NNPS': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, '-LRB-': 19,
#             'LS': 20, 'MD': 21, 'NN': 22, 'NNP': 23, 'NNPS': 24, 'NNS': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29,
#             'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33, '-RRB-': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39,
#             'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46}

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

# xpos_upos_map = {'[PAD]': '[PAD]', '[UNK]': '[UNK]', '``': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', '.': 'PUNCT',
#                  "''": 'PUNCT', '$': 'SYM', 'CC': 'CCONJ', 'CD': 'NUM', 'DATE-NNP': 'PROPN', 'DATE-NNPS': 'PROPN',
#                  'DT': 'DET', 'EX': 'PRON', 'FW': 'X', 'IN': 'ADP', 'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
#                  '-LRB-': 'PUNCT', 'LS': 'NOUN', 'MD': 'AUX', 'NN': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
#                  'NNS': 'NOUN', 'PDT': 'DET', 'POS': 'PART', 'PRP': 'PRON', 'PRP$': 'PRON', 'RB': 'ADV', 'RBR': 'ADV',
#                  'RBS': 'ADV', 'RP': 'ADP', '-RRB-': 'PUNCT', 'TO': 'PART', 'UH': 'INTJ', 'VB': 'VERB', 'VBD': 'VERB',
#                  'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'AUX', 'VBZ': 'AUX', 'WDT': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
#                  'WRB': 'ADV'}

deprel_map = {"[PAD]": 0, "[UNK]": 1, 'acl': 2, 'advcl': 3, 'advmod': 4, 'amod': 5, 'appos': 6, 'aux': 7, 'case': 8,
              'cc': 9, 'ccomp': 10, 'compound': 11, 'conj': 12, 'cop': 13, 'csubj': 14, 'det': 15, 'discourse': 16,
              'expl': 17, 'fixed': 18, 'flat': 19, 'goeswith': 20, 'iobj': 21, 'list': 22, 'mark': 23, 'nmod': 24,
              'nsubj': 25, 'nummod': 26, 'obj': 27, 'obl': 28, 'orphan': 29, 'parataxis': 30, 'punct': 31, 'root': 32,
              'vocative': 33, 'xcomp': 34}

# def map_xpos_to_upos(xpos):
#     return xpos_upos_map.get(xpos, xpos_upos_map['[UNK]'])


def get_arguments(id2tag, tag_ids, ori_example, seperate_outputs=True):
    """
    :type ori_example: nlplingo.tasks.sequence.example.SequenceExample
    """
    #ori_text = ori_example['text']
    #actual_length = len(ori_example['word'])

    # print('id2tag=', id2tag)
    # print('ori_example.words=', ori_example.words)
    # print('tag_ids=', tag_ids)

    """
    id2tag= {0: 'B-AGENT', 1: 'B-PATIENT', 2: 'I-AGENT', 3: 'I-PATIENT', 4: 'O'}
    ori_example.words= ['Betancur', 'has', '$$$', 'backed', '$$$', 'the', 'broad', 'new', 'amnesty', 'offer', 'to', 'members', 'of', 'Colombia', "'s", 'guerrilla', 'movements', 'and', 'has', 'won', 'the', 'endorsement', 'of', 'some', 'of', 'the', 'leftist', 'insurgent', 'leaders', 'by', 'offering', 'negotiations', 'and', 'studies', 'of', 'substantial', 'political', 'and', 'electoral', 'reforms', '.']
    tag_ids= tensor([0, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], device='cuda:0')
    The tag_ids are based on unmarked words, so we need to -2 from len(ori_example.words)
    """

    actual_length = len(ori_example.words) - 2
    tag_ids = tag_ids.long().data.cpu().numpy()
    tag_ids = tag_ids[: actual_length]
    tags = [id2tag[tag_id] for tag_id in tag_ids]
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
    assert len(agent_tags) == len(patient_tags)
    agents = []
    patients = []

    agent_offsets = {}
    patient_offsets = {}

    current_agent = []
    current_patient = []
    for k in range(actual_length):
        if len(current_agent) > 0 and (agent_tags[k] == 'O' or agent_tags[k].startswith('B-')):
            # start_span = ori_example['span'][current_agent[0]]
            # end_span = ori_example['span'][current_agent[-1]]
            #
            # text_span = ori_text[start_span[0]: end_span[1] + 1]
            # agents.append(text_span)
            # agent_offsets[text_span] = [start_span[0], end_span[1] + 1]

            start_char = ori_example.sentence.tokens[current_agent[0]].start_char_offset()
            end_char = ori_example.sentence.tokens[current_agent[-1]].end_char_offset()
            text_span = ori_example.sentence.get_text(start_char, end_char)
            agents.append(text_span)
            agent_offsets[text_span] = [start_char, end_char]

            current_agent = []

        if 'AGENT' in agent_tags[k]:
            current_agent.append(k)

        if len(current_patient) > 0 and (patient_tags[k] == 'O' or patient_tags[k].startswith('B-')):
            # start_span = ori_example['span'][current_patient[0]]
            # end_span = ori_example['span'][current_patient[-1]]
            #
            # text_span = ori_text[start_span[0]: end_span[1] + 1]
            # patients.append(text_span)
            # patient_offsets[text_span] = [start_span[0], end_span[1] + 1]

            start_char = ori_example.sentence.tokens[current_patient[0]].start_char_offset()
            end_char = ori_example.sentence.tokens[current_patient[-1]].end_char_offset()
            text_span = ori_example.sentence.get_text(start_char, end_char)
            patients.append(text_span)
            patient_offsets[text_span] = [start_char, end_char]

            current_patient = []

        if 'PATIENT' in patient_tags[k]:
            current_patient.append(k)

    if len(current_agent) > 0:
        # start_span = ori_example['span'][current_agent[0]]
        # end_span = ori_example['span'][current_agent[-1]]
        #
        # text_span = ori_text[start_span[0]: end_span[1] + 1]
        # agents.append(text_span)
        # agent_offsets[text_span] = [start_span[0], end_span[1] + 1]

        start_char = ori_example.sentence.tokens[current_agent[0]].start_char_offset()
        end_char = ori_example.sentence.tokens[current_agent[-1]].end_char_offset()
        text_span = ori_example.sentence.get_text(start_char, end_char)
        agents.append(text_span)
        agent_offsets[text_span] = [start_char, end_char]

    if len(current_patient) > 0:
        # start_span = ori_example['span'][current_patient[0]]
        # end_span = ori_example['span'][current_patient[-1]]
        #
        # text_span = ori_text[start_span[0]: end_span[1] + 1]
        # patients.append(text_span)
        # patient_offsets[text_span] = [start_span[0], end_span[1] + 1]

        start_char = ori_example.sentence.tokens[current_patient[0]].start_char_offset()
        end_char = ori_example.sentence.tokens[current_patient[-1]].end_char_offset()
        text_span = ori_example.sentence.get_text(start_char, end_char)
        patients.append(text_span)
        patient_offsets[text_span] = [start_char, end_char]

    if seperate_outputs:
        return agents, patients, agent_offsets, patient_offsets
    else:
        return agents + patients



class TriggerGenerator(object):
    def __init__(self, opt, xlmr_model, tokenizer, examples, docs, label_map, biw2v_map, is_eval_data=False):
        """
        :type tokenizer: python.clever.nlplingo.tasks.sequence.tokenizer.Tokenizer
        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type docs: list[nlplingo.text.text_theory.Document]
        :type label_map: dict[str, int]
        """
        # self.opt = opt
        self.tokenizer = tokenizer
        self.biw2v_map = biw2v_map
        self.xlmr_model = xlmr_model
        self.xlmr_model.eval()
        self.is_eval_data = is_eval_data
        self.encoded_data = self.encode_data(examples, docs, label_map, False)  # In Oregon code, will return: {'english': list, 'arabic': list}
        #self.num_examples = len(self.encoded_data['english'] + self.encoded_data['arabic'])
        self.num_examples = len(self.encoded_data)
        self.data_batches = self.create_batches(self.encoded_data, opt['batch_size'])
        self.num_batches = len(self.data_batches)

    def encode_trigger_example(self, example, doc, label_map):
        """modeled after: python.clever.event_models.uoregon.models.pipeline._01.iterators.EDIterator.encode_example

        In Oregon's code:
        #### Before encoding
        'triggers': [[21, 22], [34, 35]]
        'event-types': ['harmful|material', 'neutral|material']
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

        :type example: nlplingo.tasks.sequence.example.SequenceExample
        :type doc: nlplingo.text.text_theory.Document
        :type label_map: dict[str, int]
        """
        word_list = example.words
        ED_labels = [label_map[label_string] for label_string in example.labels]

        xlmr_ids, input_mask, label_ids, retrieve_ids, subwords = self.tokenizer.tokenize_to_ids(word_list, ED_labels)
        #xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list)
        xpos_ids = [xpos_map.get(token.pos_tag, xpos_map['[UNK]']) for token in example.sentence.tokens]
        upos_ids = [upos_map.get(token.pos_tag_alternate, upos_map['[UNK]']) for token in example.sentence.tokens]
        #upos_ids = [upos_map[map_xpos_to_upos(token.pos_tag)] for token in example.sentence.tokens]
        #print('len(doc.adj_mats)=', len(doc.adj_mats))
        #print('example.sentence_index=', example.sentence_index)

        head_ids = [token.dep_relations[0].connecting_token_index for token in example.sentence.tokens]
        dep_rel_ids = [deprel_map.get(token.dep_relations[0].dep_name, deprel_map['[UNK]']) for token in example.sentence.tokens]
        # head_ids = doc.adj_mats[example.sentence_index][0]
        # dep_rel_ids = doc.adj_mats[example.sentence_index][1]

        ner_list = ['O'] * len(word_list)
        ner_ids = [ner_map.get(ner) for ner in ner_list]
        lang_weight = 1.0
        #biw2v_ids = [1] * len(word_list)    # TODO
        biw2v_ids = [self.biw2v_map.get(word.lower(), self.biw2v_map[UNK_TOKEN]) for word in word_list]

        print('example.words=', example.words)
        print('example.labels=', example.labels)
        #print('subwords=', subwords)
        print('xlmr_ids=', xlmr_ids)
        #print('input_mask=', input_mask)
        #print('label_ids=', label_ids)
        #print('biw2v_ids=', biw2v_ids)
        print('retrieve_ids=', retrieve_ids)
        #print('upos_ids=', upos_ids)
        #print('xpos_ids=', xpos_ids)
        #print('head_ids=', head_ids)
        #print('dep_rel_ids=', dep_rel_ids)
        #print('ner_ids=', ner_ids)
        #print('lang_weight=', lang_weight)
        #print('ED_labels=', ED_labels)
        #sys.exit(0)
        return xlmr_ids, input_mask, label_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, lang_weight, ED_labels

    def encode_data(self, examples, docs, label_map, is_eval_data):
        """ For each example in examples, "generate features" to become encoded_ex. Then shuffle if not eval data

        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type docs: list[nlplingo.text.text_theory.Document]
        :type is_eval_data: bool
        """
        docid_to_doc = dict()
        for doc in docs:
            docid_to_doc[doc.docid] = doc

        encoded_examples = []
        for example in examples:
            encoded_ex = self.encode_trigger_example(example, docid_to_doc[example.docid], label_map)
            encoded_examples.append(encoded_ex)

        if not is_eval_data:
            encoded_examples = shuffle_list(encoded_examples)

        return encoded_examples

    def create_batches(self, examples, batch_size):
        """ modified from uoregon code, where I assume all examples are English
        If:        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Then batches = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        """
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
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
        #xlmr_ids = do_padding(batch[0], batch_size)

        #xlmr_ids = do_padding(batch[0], batch_size, pad_id=self.tokenizer.pad_token)
        xlmr_ids = do_padding(batch[0], batch_size)
        input_mask = do_padding(batch[1], batch_size)
        label_ids = do_padding(batch[2], batch_size, pad_id=self.tokenizer.pad_token_label_id)

        biw2v_ids = do_padding(batch[3], batch_size)
        # ***********************************************
        retrieve_ids = do_padding(batch[4], batch_size)
        upos_ids = do_padding(batch[5], batch_size)
        xpos_ids = do_padding(batch[6], batch_size)

        head_ids = do_padding(batch[7], batch_size)
        deprel_ids = do_padding(batch[8], batch_size)
        ner_ids = do_padding(batch[9], batch_size)
        lang_weights = torch.Tensor(batch[10])

        ED_labels = do_padding(batch[11], batch_size)

        # If:   retrieve_ids = torch.LongTensor([1, 2, 3, 0, 5])
        # Then: torch.eq(retrieve_ids, 0) produces: tensor([False, False, False,  True, False])
        pad_masks = torch.eq(retrieve_ids, 0)

        print('xlmr_ids=', xlmr_ids)
        print('input_mask=', input_mask)
        print('retrieve_ids=', retrieve_ids)
        print('ED_labels=', ED_labels)
        print('pad_masks=', pad_masks)
        return (
            xlmr_ids, input_mask, label_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights,
            ED_labels,
            pad_masks)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)   # yield a batch

    def shuffle_batches(self):
        indices = list(range(len(self.data_batches)))
        random.shuffle(indices)
        self.data_batches = [self.data_batches[i] for i in indices]


class ArgumentGenerator:
    def __init__(self, opt, xlmr_model, examples, docs, label_map, biw2v_map, is_eval_data=False):

        # self.opt = opt
        self.biw2v_map = biw2v_map
        self.xlmr_model = xlmr_model
        self.xlmr_model.eval()
        self.is_eval_data = is_eval_data
        # self.data_path = data_path

        self.id2ori_example = {}
        self.id2tag = dict([(v, k) for k, v in ARGUMENT_TAG_MAP.items()])

        self.encoded_data = self.encode_data(examples, docs, label_map, False)
        #self.num_examples = len(self.encoded_data['english'] + self.encoded_data['arabic'])
        self.num_examples = len(self.encoded_data)
        self.data_batches = self.create_batches(self.encoded_data, opt['batch_size'])
        self.num_batches = len(self.data_batches)

    def encode_argument_example(self, example, doc, label_map):
        """ modeled after: python.clever.event_models.uoregon.models.pipeline._01.iterators.ArgumentIterator.encode_example
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

        # example.words is marked up, like so: w0 w1 w2 $$$ w3 w4 $$$$ w5 w6,
        # where '$$$' surrounds the anchor span
        # We find the indices of the markers. In the above example, we should get: [3, 6]
        marker_indices = find_token_indices_of_markers(example.words)
        assert len(marker_indices) == 2

        words_without_markers = remove_tokens_at_indices(example.words, marker_indices)
        labels_without_markers = remove_tokens_at_indices(example.labels, marker_indices)

        anchor_start_token_index = marker_indices[0]
        anchor_end_token_index = marker_indices[1] - 2
        # Why is there a -2 in the above? Assume that you have the following marked up word sequence:
        # 0 1 2  3  4 5  6  7 8  (token indices of marked up sequence)
        # 0 1 2 $$$ 3 4 $$$ 5 6  (original token indices)
        # On the above:
        # * marker_indices[0] == 3
        # * marker_indices[1] == 6 (you need to minus by 2, to get 4, which is the original token index)
        trigger_word = words_without_markers[anchor_start_token_index]

        word_list = words_without_markers
        entity_tags = [label_map[label_string] for label_string in labels_without_markers]

        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list, trigger_word) # 'trigger_word' is raw text string of the trigger
        xpos_ids = [xpos_map.get(token.pos_tag, xpos_map['[UNK]']) for token in example.sentence.tokens]
        upos_ids = [upos_map.get(token.pos_tag_alternate, upos_map['[UNK]']) for token in example.sentence.tokens]
        #upos_ids = [upos_map[map_xpos_to_upos(token.pos_tag)] for token in example.sentence.tokens]

        head_ids = [token.dep_relations[0].connecting_token_index for token in example.sentence.tokens]
        dep_rel_ids = [deprel_map.get(token.dep_relations[0].dep_name, deprel_map['[UNK]']) for token in example.sentence.tokens]
        # head_ids = doc.adj_mats[example.sentence_index][0]
        # dep_rel_ids = doc.adj_mats[example.sentence_index][1]

        ner_list = ['O'] * len(word_list)
        ner_ids = [ner_map.get(ner) for ner in ner_list]
        lang_weight = 1.0
        #biw2v_ids = [1] * len(word_list)  # TODO
        biw2v_ids = [self.biw2v_map.get(word.lower(), self.biw2v_map[UNK_TOKEN]) for word in word_list]
        #######

        # word_list = example['word']
        # upos_list = example['upos']
        # xpos_list = example['xpos']
        # head_list = example['head']
        # dep_rel_list = example['dep_rel']
        # ner_list = example['ner']
        # lang_weight = 1.0 if opt['co_train_lambda'] == 0 or example['lang'] == 'english' else opt['co_train_lambda']
        # # *****************************
        # trigger_toks = example['trigger']
        #
        # # get raw text string of the trigger
        # trigger_word = example['text'][
        #                example['span'][int(trigger_toks[0])][0]: example['span'][int(trigger_toks[-1])][1] + 1]
        #
        # xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list, trigger_word)
        # # ****** biw2v ************
        # biw2v_ids = [biw2v_map.get(word.lower(), biw2v_map[UNK_TOKEN]) for word in word_list]
        # # *****************************
        # upos_ids = [upos_map.get(upos, upos_map[UNK_TOKEN]) for upos in upos_list]
        # xpos_ids = [xpos_map.get(xpos, xpos_map[UNK_TOKEN]) for xpos in xpos_list]
        # head_ids = head_list
        # dep_rel_ids = [deprel_map.get(dep_rel, deprel_map[UNK_TOKEN]) for dep_rel in dep_rel_list]
        # ner_ids = [ner_map.get(ner, ner_map[UNK_TOKEN]) for ner in ner_list]
        #
        # entity_tags = get_bio_tags(
        #     agent_extents=example['agents'],
        #     patient_extents=example['patients'],
        #     word_list=word_list
        # )
        # trigger_tok = example['trigger'][0]
        #
        eid = len(self.id2ori_example)
        self.id2ori_example[eid] = example
        return xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, lang_weight, anchor_start_token_index, entity_tags, eid

    def encode_data(self, examples, docs, label_map, is_eval_data):
        """ For each example in examples, "generate features" to become encoded_ex. Then shuffle if not eval data

        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type docs: list[nlplingo.text.text_theory.Document]
        :type is_eval_data: bool
        """
        docid_to_doc = dict()
        for doc in docs:
            docid_to_doc[doc.docid] = doc

        encoded_examples = []
        for example in examples:
            encoded_ex = self.encode_argument_example(example, docid_to_doc[example.docid], label_map)
            encoded_examples.append(encoded_ex)

        if not is_eval_data:
            encoded_examples = shuffle_list(encoded_examples)

        return encoded_examples

    def create_batches(self, examples, batch_size):
        """ modified from uoregon code, where I assume all examples are English
        If:        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Then batches = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        """
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
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

        lens = [len(x) for x in batch[0]]  # batch[0] is list[xlmr_ids], so this gives: [len(xlmr_ids) for each example]
        batch, _ = sort_all(batch, lens)  # sort elements in batch by decreasing order of their len

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


class PipelineGenerator(object):
    """
    eval -> PipelineTrainer -> PipelineIterator
    """
    def __init__(self, xlmr_model, examples, docs):
        # e.g. data_path = datapoints/data/arabic-abstract-sample/arabic-abstract-sample.pipeline.test (this is already parsed by stanfordnlp)
        #print('Using {} for pipeline iterator...'.format(data_path))

        #self.opt = opt
        self.xlmr_model = xlmr_model  # This is the XLMRModel.from_pretrained
        self.xlmr_model.eval()
        #self.data_path = data_path

        self.id2ori_example = {}

        self.encoded_data = self.encode_data(examples, docs)
        self.num_examples = len(self.encoded_data)

        self.data_batches = self.create_batches(self.encoded_data, 8)
        self.num_batches = len(self.data_batches)

    def encode_pipeline_example(self, example, doc):
        #example['norm2ori_offsetmap'] = {int(k): v for k, v in example['norm2ori_offsetmap'].items()}  # TODO

        word_list = example.words
        xlmr_ids, retrieve_ids = xlmr_tokenizer.get_token_ids(self.xlmr_model, word_list)
        biw2v_ids = [1] * len(word_list)  # TODO
        # biw2v_ids = [biw2v_map.get(word.lower(), biw2v_map[UNK_TOKEN]) for word in word_list]
        xpos_ids = [xpos_map.get(token.pos_tag, xpos_map['[UNK]']) for token in example.sentence.tokens]
        upos_ids = [upos_map.get(token.pos_tag_alternate, upos_map['[UNK]']) for token in example.sentence.tokens]
        # upos_ids = [upos_map[map_xpos_to_upos(token.pos_tag)] for token in example.sentence.tokens]

        head_ids = [token.dep_relations[0].connecting_token_index for token in example.sentence.tokens]
        dep_rel_ids = [deprel_map.get(token.dep_relations[0].dep_name, deprel_map['[UNK]']) for token in example.sentence.tokens]
        # head_ids = doc.adj_mats[example.sentence_index][0]
        # dep_rel_ids = doc.adj_mats[example.sentence_index][1]

        ner_list = ['O'] * len(word_list)
        ner_ids = [ner_map.get(ner) for ner in ner_list]

        eid = len(self.id2ori_example)
        self.id2ori_example[eid] = example
        return xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, dep_rel_ids, ner_ids, eid

    def encode_data(self, examples, docs):
        """
        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        # if self.opt['input_lang'] != 'english' and self.opt['use_dep2sent']:
        #     data = read_pickle(self.data_path)  # alternative data permutated by dep2sent model
        # else:
        #     data = read_json(self.data_path)

        docid_to_doc = dict()
        for doc in docs:
            docid_to_doc[doc.docid] = doc

        encoded_examples = []
        for example in examples:
            encoded_ex = self.encode_pipeline_example(example, docid_to_doc[example.docid])
            encoded_examples.append(encoded_ex)

        return encoded_examples

    def create_batches(self, examples, batch_size):
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
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
