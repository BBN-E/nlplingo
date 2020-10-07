from transformers import BertTokenizer
import os, sys, logging
import time
logger = logging.getLogger(__name__)
import argparse
import glob
import re

# filter out training data that can't be used for mention pooling.
def tokenize_lengths(item, tokenizer, max_length, mention_pool=True):
    """
    Args:
        item: data instance containing 'text' / 'token', 'h' and 't'
    Return:
        Name of the relation of the sentence
    """
    # Sentence -> token
    if 'text' in item:
        sentence = item['text']
        is_token = False
    else:
        sentence = item['token']
        is_token = True
    pos_head = item['h']['pos']
    pos_tail = item['t']['pos']
    mask_entity = False
    if not is_token:
        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        sent0 = tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = tokenizer.tokenize(sentence[pos_max[1]:])
        if mask_entity:
            ent0 = ['[unused4]']
            ent1 = ['[unused5]']
            if rev:
                ent0 = ['[unused5]']
                ent1 = ['[unused4]']
        pos_head = [len(sent0), len(sent0) + len(ent0)]
        pos_tail = [
            len(sent0) + len(ent0) + len(sent1),
            len(sent0) + len(ent0) + len(sent1) + len(ent1)
        ]
        if rev:
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2
    else:
        tokens = sentence

    head_indices = []
    tail_indices = []
    # Token -> index
    re_tokens = ['[CLS]']
    cur_pos = 0
    for token in tokens:
        token = token.lower()
        if not mention_pool:
            if cur_pos == pos_head[0] and not mask_entity:
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0] and not mask_entity:
                re_tokens.append('[unused1]')

        if mention_pool:
            if cur_pos == pos_head[0]:
                head_indices.append(len(re_tokens))
            if cur_pos == pos_tail[0]:
                tail_indices.append(len(re_tokens))

        if is_token:
            re_tokens += tokenizer.tokenize(token)
        else:
            re_tokens.append(token)

        if not mention_pool:
            if cur_pos == pos_head[1] - 1 and not mask_entity:
                re_tokens.append('[unused2]')

            if cur_pos == pos_tail[1] - 1 and not mask_entity:
                re_tokens.append('[unused3]')

        if mention_pool:
            if cur_pos == pos_head[1] - 1:
                head_indices.append(len(re_tokens))
            if cur_pos == pos_tail[1] - 1:
                tail_indices.append(len(re_tokens))
        cur_pos += 1
    re_tokens.append('[SEP]')
    # print('head_indices', head_indices)
    # print('tail_indices', tail_indices)
    indexed_tokens = tokenizer.convert_tokens_to_ids(re_tokens)
    avai_len = len(indexed_tokens)
    if mention_pool:
        assert (len(head_indices) == 2)
        assert (len(tail_indices) == 2)
    if head_indices[1] >= max_length or (tail_indices[1] >= max_length):
        return False
    else:
        return True

def mention_pool_filter(args):
    file_prefix = args.file_prefix
    files = glob.glob(args.txt_file_dir + "/" + file_prefix + "_*")

    for txt_file in files:
        base = os.path.basename(txt_file)
        base = os.path.splitext(base)[0]
        filtered_file = args.output_directory + '/' + base + '_filtered.json'
        tokenizer = BertTokenizer.from_pretrained('/nfs/raid87/u10/shared/Hume/common/event_event_relation/models/opennre-data/pretrain/bert-base-uncased/')
        max_length = 128

        f = open(txt_file)
        deleted_indices = []
        objs = []
        start = time.time()
        for i, line in enumerate(f.readlines()):
            line = line.rstrip()
            if len(line) > 0:
                obj = eval(line)
                if tokenize_lengths(obj, tokenizer, max_length):
                    objs.append(obj)
                else:
                    deleted_indices.append(i)
        end = time.time()
        print('Tokenizing took', end - start, 'seconds')

        start = time.time()
        with open(filtered_file, 'w') as fp:
            for obj in objs:
                fp.write(str(obj) + "\n")
        end = time.time()
        print('Writing took', end - start, 'seconds')

def mention_pool_filter_from_memory(triplets, output_file):
    tokenizer = BertTokenizer.from_pretrained(
        '/nfs/raid87/u10/shared/Hume/common/event_event_relation/models/opennre-data/pretrain/bert-base-uncased/')

    max_length = 128

    deleted_indices = []
    objs = []
    start = time.time()
    for i, obj in enumerate(triplets):
        if tokenize_lengths(obj, tokenizer, max_length):
            objs.append(obj)
        else:
            deleted_indices.append(i)
    end = time.time()
    print('Tokenizing took', end - start, 'seconds')

    start = time.time()
    with open(output_file, 'w') as fp:
        for obj in objs:
            fp.write(str(obj) + "\n")
    end = time.time()
    print('Writing took', end - start, 'seconds')

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('txt_file_dir')
    arg_parser.add_argument('file_prefix')
    arg_parser.add_argument('output_directory')
    _args = arg_parser.parse_args(args_list)
    return _args

if __name__ == '__main__':
    log_format = '[%(asctime)s] {P%(process)d:%(module)s:%(lineno)d} %(levelname)s - %(message)s'
    try:
        logging.basicConfig(level=logging.getLevelName(os.environ.get('LOGLEVEL', 'INFO').upper()),
                            format=log_format)
    except ValueError as e:
        logging.error(
            "Unparseable level {}, will use default {}.".format(os.environ.get('LOGLEVEL', 'INFO').upper(),
                                                                logging.root.level))
        logging.basicConfig(format=log_format)
    mention_pool_filter(parse(sys.argv[1:]))