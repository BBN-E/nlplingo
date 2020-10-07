import os, json
import numpy as np
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *

#MODEL_DIR = os.path.dirname(os.path.realpath(__file__))    # <==

DATAPATH_MAP = {
    'trigger': {
        'train': 'data/{}/BETTER_sample.trigger.train',
        'dev': 'data/{}/BETTER_sample.trigger.dev'
    },
    'entity': {
        'train': 'data/{}/BETTER_sample.entity.train',
        'dev': 'data/{}/BETTER_sample.entity.dev'
    },
    'event': {
        'train': 'data/{}/BETTER_sample.event.train',
        'dev': 'data/{}/BETTER_sample.event.dev'
    },
    'combined_task': {
        'test': 'data/{}/BETTER_sample.combined_task.test'
    }
}


AGENT_MAP = {
    'NOT-AGENT': 0,
    'AGENT': 1,
}

PATIENT_MAP = {
    'NOT-PATIENT': 0,
    'PATIENT': 1
}

PAD = "<PAD>"  # padding
SOS = "<SOS>"  # start of sequence
EOS = "<EOS>"  # end of sequence

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

ARGUMENT_TAG_MAP = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2,
                  'O': 3, 'B-AGENT': 4, 'I-AGENT': 5,
                  'B-PATIENT': 6, 'I-PATIENT': 7,
                    # ********** conflict cases **********
                  'B-AGENT|B-PATIENT': 8, 'B-AGENT|I-PATIENT': 9, 'I-AGENT|B-PATIENT': 10, 'I-AGENT|I-PATIENT': 11
                    }

WINDOW_SIZE = 50

NUM_DISTANCES = int(np.sum([1,  # padding token
                            1,  # out of left context
                            1,  # out of right context
                            WINDOW_SIZE,  # left context
                            1,  # trigger word
                            WINDOW_SIZE  # right context
                            ]))

IMPACT_MAP = {
    UNKNOWN_EVENT_KEY: 0,
    HARMFUL_KEY: 1,
    HELPFUL_KEY: 2,
    NEUTRAL_KEY: 3
}
EFFECT_MAP = {
    UNKNOWN_EVENT_KEY: 0,
    MATERIAL_KEY: 1,
    VERBAL_KEY: 2,
    MATERIAL_VERBAL_KEY: 3
}
EVENT_MAP = {
    '{}|{}'.format(UNKNOWN_EVENT_KEY, UNKNOWN_EVENT_KEY): 0,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, MATERIAL_KEY): 1,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, VERBAL_KEY): 2,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, MATERIAL_VERBAL_KEY): 3,
    '{}|{}'.format(HARMFUL_KEY, UNKNOWN_EVENT_KEY): 4,
    '{}|{}'.format(HARMFUL_KEY, MATERIAL_KEY): 5,
    '{}|{}'.format(HARMFUL_KEY, VERBAL_KEY): 6,
    '{}|{}'.format(HARMFUL_KEY, MATERIAL_VERBAL_KEY): 7,
    '{}|{}'.format(HELPFUL_KEY, UNKNOWN_EVENT_KEY): 8,
    '{}|{}'.format(HELPFUL_KEY, MATERIAL_KEY): 9,
    '{}|{}'.format(HELPFUL_KEY, VERBAL_KEY): 10,
    '{}|{}'.format(HELPFUL_KEY, MATERIAL_VERBAL_KEY): 11,
    '{}|{}'.format(NEUTRAL_KEY, UNKNOWN_EVENT_KEY): 12,
    '{}|{}'.format(NEUTRAL_KEY, MATERIAL_KEY): 13,
    '{}|{}'.format(NEUTRAL_KEY, VERBAL_KEY): 14,
    '{}|{}'.format(NEUTRAL_KEY, MATERIAL_VERBAL_KEY): 15
}
