import os
import sys
import argparse
import json
from collections import defaultdict

import serifxml3 as serifxml

from nlplingo.common.utils import IntPair
from nlplingo.text.text_span import Token
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.embeddings.word_embeddings import load_embeddings
from nlplingo.tasks.eventtrigger.generator import EventTriggerFeatureGenerator

valid_labels = {'harmful.both', 'harmful.material', 'harmful.verbal', 'helpful.both', 'helpful.material',
                'helpful.verbal', 'neutral.both', 'neutral.material', 'neutral.verbal',
                'helpful.unk', 'harmful.unk', 'neutral.unk'}

def transform_sentence_labels_to_BIO(token_labels):
    # first, transform the token_labels to BIO format
    for i in range(0, len(token_labels)):
        token_labels[i] = token_labels[i].lower()
        if token_labels[i] not in valid_labels:
            token_labels[i] = 'none'            

    for i in range(len(token_labels)-1, 0, -1):
        if token_labels[i] == 'none':
            token_labels[i] = 'O'
        else:
            if token_labels[i] == token_labels[i-1]:
                token_labels[i] = 'I-{}'.format(token_labels[i])
            else:
                token_labels[i] = 'B-{}'.format(token_labels[i])

    if token_labels[0] == 'none':
        token_labels[0] = 'O'
    else:
        token_labels[0] = 'B-{}'.format(token_labels[0])

    return token_labels


def lingodoc_trigger_to_BIO(doc):
    """
    :type doc: nlplingo.text.text_theory.Document
    """

    ret = []
    for sentence in doc.sentences:
        token_labels = []
        for token_index, token in enumerate(sentence.tokens):
            token_labels.append(EventTriggerFeatureGenerator.get_event_type_of_token(token, sentence))

        bio_labels = transform_sentence_labels_to_BIO(token_labels)

        token_bio = []
        for k, v in zip(sentence.tokens, bio_labels):
            token_bio.append('{} {}'.format(k.text, v))

        ret.append('\n'.join(token_bio))

    return ret


def serif_trigger_to_conll_format(params, word_embeddings):
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    train_data = []
    for doc in train_docs:
        train_data.append('\n\n'.join(lingodoc_trigger_to_BIO(doc)))
    with open(os.path.join(params['output_dir'], 'train.txt'), 'w', encoding='utf-8') as o:
        o.write('\n\n'.join(train_data))

    dev_data = []
    for doc in dev_docs:
        dev_data.append('\n\n'.join(lingodoc_trigger_to_BIO(doc)))
    with open(os.path.join(params['output_dir'], 'dev.txt'), 'w', encoding='utf-8') as o:
        o.write('\n\n'.join(dev_data))

    test_data = []
    for doc in test_docs:
        test_data.append('\n\n'.join(lingodoc_trigger_to_BIO(doc)))
    with open(os.path.join(params['output_dir'], 'test.txt'), 'w', encoding='utf-8') as o:
        o.write('\n\n'.join(test_data))


def serif_argument_to_conll_format(params):
    pass


def extract_trigger_BIO_from_serifxml(filelist, output_dir):
    with open(filelist, 'r', encoding='utf-8') as f:
        filepaths = [line.rstrip() for line in f.readlines()]

    text_bio = []
    raw_text_bio = []
    for filepath in filepaths:
        serif_doc = serifxml.Document(filepath)
        docid = serif_doc.docid

        for st_index, sentence in enumerate(serif_doc.sentences):
            offsets_eventtypes = []
            for event_mention in sentence.event_mention_set:
                mention_id = event_mention.id
                event_type = event_mention.event_type

                anchor = event_mention.anchor_node
                start = anchor.start_token.start_edt
                end = anchor.end_token.end_edt
                text = anchor.text
                offsets_eventtypes.append((start, end, event_type))

            # construct serif tokens
            tokens = []
            st = sentence.sentence_theories[0]
            root = st.parse.root
            for i, t in enumerate(root.terminals):
                t_text = t.text
                t_start = t.start_token.start_edt
                t_end = t.end_token.end_edt
                t_pos_tag = t.parent.tag
                tokens.append(Token(IntPair(t_start, t_end), i, t_text, lemma=None, pos_tag=t_pos_tag))

            # construct BIO according to serif tokens
            tokens_eventtypes = []
            for i, token in enumerate(tokens):
                label = 'None'
                for (start, end, event_type) in offsets_eventtypes:
                    if start <= token.start_char_offset() and token.end_char_offset() <= end:
                        label = event_type
                        break
                tokens_eventtypes.append(label)
            tokens_bio = transform_sentence_labels_to_BIO(tokens_eventtypes)
            assert len(tokens) == len(tokens_bio)

            tokens_text_bio = []
            for k, v in zip(tokens, tokens_bio):
                tokens_text_bio.append('{} {}'.format(k.text, v))
            text_bio.append('\n'.join(tokens_text_bio))
            #=======================================================

            # construct raw tokens
            raw_tokens = []
            current_start = sentence.start_edt
            for i, token_text in enumerate(sentence.text.split()):
                current_end = current_start + len(token_text) - 1
                raw_tokens.append(Token(IntPair(current_start, current_end), i, token_text, lemma=None, pos_tag=None))
                current_start = current_end + 2

            # construct BIO according to raw tokens
            raw_tokens_eventtypes = []
            for i, token in enumerate(raw_tokens):
                label = 'None'
                for (start, end, event_type) in offsets_eventtypes:
                    if (start <= token.start_char_offset() and token.end_char_offset() <= end) or \
                        (token.start_char_offset() <= start and start <= token.end_char_offset()) or \
                        (token.end_char_offset() <= end and end <= token.end_char_offset()):
                        label = event_type
                        break
                raw_tokens_eventtypes.append(label)
            raw_tokens_bio = transform_sentence_labels_to_BIO(raw_tokens_eventtypes)
            assert len(raw_tokens) == len(raw_tokens_bio)

            raw_tokens_text_bio = []
            for k, v in zip(raw_tokens, raw_tokens_bio):
                raw_tokens_text_bio.append('{} {}'.format(k.text, v))
            raw_text_bio.append('\n'.join(raw_tokens_text_bio))

    with open(os.path.join(output_dir, 'text.bio'), 'w', encoding='utf-8') as o:
        o.write('\n\n'.join(text_bio))

    with open(os.path.join(output_dir, 'raw_text.bio'), 'w', encoding='utf-8') as o:
        o.write('\n\n'.join(raw_text_bio))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)  # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)

    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)

    if args.mode == 'serif_trigger_to_conll_format':
        serif_trigger_to_conll_format(params, embeddings)
    elif args.mode == 'serif_argument_to_conll_format':
        serif_argument_to_conll_format(params, embeddings)
    elif args.mode == 'extract_trigger_BIO_from_serifxml':
        extract_trigger_BIO_from_serifxml(params['serifxml_filelist'], params['output_dir'])

if __name__ == "__main__":
    main()
