import logging
logger = logging.getLogger(__name__)

supported_pytorch_models = {
    'event-event-relation_cnn-embedded',
    'event-argument_bert-mention',
    'entity-entity-relation_bert-mention'
}

# these global variables (which are in poor taste) are needed to deal with PyTorch DataLoader's need for
# a collate function that does not accept convenient configuration variables
GLOBAL_MAX_SENT_LENGTH = 128
GLOBAL_BLANK_ID = -1

def change_global_max_sent_length(sent_length):
    global GLOBAL_MAX_SENT_LENGTH
    GLOBAL_MAX_SENT_LENGTH = sent_length
    logging.info("Changed GLOBAL_MAX_SENT_LENGTH to: %s", GLOBAL_MAX_SENT_LENGTH)

def get_global_max_sent_length():
    global GLOBAL_MAX_SENT_LENGTH
    return GLOBAL_MAX_SENT_LENGTH

def change_global_blank_id(blank_id):
    global GLOBAL_BLANK_ID
    GLOBAL_BLANK_ID = blank_id
    print("Changed GLOBAL_BLANK_ID to:", GLOBAL_BLANK_ID)

def get_global_blank_id():
    global GLOBAL_BLANK_ID
    return GLOBAL_BLANK_ID
