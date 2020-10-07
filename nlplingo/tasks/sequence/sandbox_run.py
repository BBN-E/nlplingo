import logging
import argparse
from collections import defaultdict
import json

import serifxml3 as serifxml

from nlplingo.tasks.sequence.run import get_frame_annotations_from_bp
from nlplingo.annotation.ingestion import prepare_docs


def labeled_text_frame_tostring(annotation):
    """
    :type annotation: nlplingo.text.text_span.LabeledTextFrame
    """
    ret = []

    for span in annotation.anchor_spans:
        lines = []
        anchor_texts = []
        anchor_texts.append('{}[{}]'.format(span.label, span.text))
        lines.append('anchors: {}'.format(' ||| '.join(anchor_texts)))
        args_by_roles = defaultdict(list)
        for argument in annotation.argument_spans:
            args_by_roles[argument.label].append(argument)
        for role in args_by_roles:
            arg_text = ' ||| '.join(sorted(set('[{}]'.format(arg.text) for arg in args_by_roles[role])))    # remove duplicates to make automatic comparison easier
            lines.append('{}: {}'.format(role, arg_text))
        ret.append('\n'.join(lines))
    return ret


def nlplingo_event_tostring(event):
    ret = []
    anchor_texts = []
    for anchor in event.anchors:
        anchor_texts.append('{}[{}]'.format(anchor.label, anchor.text))
    ret.append('anchors: {}'.format(' ||| '.join(anchor_texts)))
    args_by_roles = defaultdict(list)
    for argument in event.arguments:
        args_by_roles[argument.label].append(argument)
    for role in args_by_roles:
        arg_text = ' ||| '.join(sorted(set('[{}]'.format(arg.text) for arg in args_by_roles[role])))        # remove duplicates to make automatic comparison easier
        ret.append('{}: {}'.format(role, arg_text))
    return '\n'.join(ret)


def serif_event_tostring(event_mention, serif_sent):
    ret = []

    if event_mention.semantic_phrase_start is not None and event_mention.semantic_phrase_end is not None:
        start_index = int(event_mention.semantic_phrase_start)
        end_index = int(event_mention.semantic_phrase_end)
        start = serif_sent.token_sequence[start_index:end_index+1][0].start_edt
        end = serif_sent.token_sequence[start_index:end_index+1][-1].end_edt
        text = serif_sent.get_original_text_substring(start, end)
        ret.append('anchors: {}[{}]'.format(event_mention.event_type, text))
    else:
        ret.append('anchors: {}[{}]'.format(event_mention.event_type, event_mention.anchor_node.text))

    args_by_roles = defaultdict(list)
    for argument in event_mention.arguments:
        args_by_roles[argument.role].append(argument)
    for role in args_by_roles:
        arg_text = ' ||| '.join(sorted(set('[{}]'.format(arg.value.text) for arg in args_by_roles[role])))  # remove duplicates to make automatic comparison easier
        ret.append('{}: {}'.format(role, arg_text))
    return '\n'.join(ret)


def compare_bp_with_lingodoc(params):
    """
    This compares BP annotations with Serif annotation (after they have been ingested into NLPLingo)
    """
    annotations = get_frame_annotations_from_bp(params['bp_file'])
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""

    docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    diff_docids = set()
    for doc in docs:
        assert doc.docid in annotations
        doc_annotation = annotations[doc.docid]
        assert len(doc.sentences) == len(doc_annotation)

        for i, sentence in enumerate(doc.sentences):
            print(doc.docid, sentence.text)

            # annotations from Serif
            serif_strings = []
            for event in sentence.events:
                print('## SERIF event')
                serif_string = nlplingo_event_tostring(event)
                print(serif_string)
                serif_strings.append(serif_string)

            # annotations from BP
            bp_strings = []
            sentence_annotation = doc_annotation[i]
            for annotation in sentence_annotation:
                print('## BP event')
                bp_string = labeled_text_frame_tostring(annotation)
                print('\n'.join(bp_string))
                bp_strings.extend(bp_string)
            print('')

            if '\n'.join(sorted(serif_strings)) != '\n'.join(sorted(bp_strings)):
                diff_docids.add(doc.docid)

    print('#### {} docids with differences'.format(str(len(diff_docids))))
    for docid in sorted(diff_docids):
        print(docid)


def get_diff(strings1, strings2):
    lines1 = sorted(strings1)
    lines2 = sorted(strings2)

    discard1 = set()
    discard2 = set()

    for i, line1 in enumerate(lines1):
        for j, line2 in enumerate(lines2):
            if line1 == line2:
                discard1.add(i)
                discard2.add(j)
                break
        if i in discard1:
            continue

    ret1 = []
    ret2 = []
    for i, line in enumerate(lines1):
        if i not in discard1:
            ret1.append(line)
    for i, line in enumerate(lines2):
        if i not in discard2:
            ret2.append(line)

    return (ret1, ret2)


def compare_bp_with_serifxml(params):
    """
    This compares BP annotations with SerifXML EventMention
    """
    annotations = get_frame_annotations_from_bp(params['bp_file'])
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""

    docs = []
    with open(params['data']['train']['filelist'], 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('SERIF:'):
                filepath = line.strip()[6:]
                docs.append(serifxml.Document(filepath))

    diff_docids = dict()
    for doc in docs:
        """:type: serifxml.Document"""

        assert doc.docid in annotations
        doc_annotation = annotations[doc.docid]
        assert len(doc.sentences) == len(doc_annotation)

        for i, sentence in enumerate(doc.sentences):
            print(doc.docid, sentence.text)

            # annotations from Serif
            serif_strings = []
            for event_mention in sentence.event_mention_set:
                print('## SERIF event')
                serif_string = serif_event_tostring(event_mention, sentence)
                print(serif_string)
                serif_strings.append(serif_string)

            # annotations from BP
            bp_strings = []
            sentence_annotation = doc_annotation[i]
            for annotation in sentence_annotation:
                print('## BP event')
                bp_string = labeled_text_frame_tostring(annotation)
                print('\n'.join(bp_string))
                bp_strings.extend(bp_string)
            print('')

            if '\n'.join(sorted(serif_strings)) != '\n'.join(sorted(bp_strings)):
                strings1, strings2 = get_diff(serif_strings, bp_strings)
                diff_docids[doc.docid] = '#### SERIF\n' + '\n========\n'.join(strings1) + '\n#### BP\n' + '\n========\n'.join(strings2)

    print('#### {} docids with differences'.format(str(len(diff_docids))))
    for docid in sorted(diff_docids):
        print('******** ' + docid)
        print(diff_docids[docid])


def compare_serifxml_with_lingodoc(params):
    docs = []
    with open(params['data']['train']['filelist'], 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('SERIF:'):
                filepath = line.strip()[6:]
                docs.append(serifxml.Document(filepath))

    lingo_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""
    id_to_lingodoc = dict()
    for doc in lingo_docs:
        id_to_lingodoc[doc.docid] = doc

    diff_docids = dict()
    for doc in docs:
        """:type: serifxml.Document"""

        assert doc.docid in id_to_lingodoc
        lingo_doc = id_to_lingodoc[doc.docid]

        assert len(doc.sentences) == len(lingo_doc.sentences)

        for i, sentence in enumerate(doc.sentences):
            print(doc.docid, sentence.text)

            serif_strings = []
            # annotations from Serif
            for event_mention in sentence.event_mention_set:
                print('## SERIF event')
                serif_string = serif_event_tostring(event_mention, sentence)
                print(serif_string)
                serif_strings.append(serif_string)

            lingo_strings = []
            # annotations from lingo_doc
            lingo_sentence = lingo_doc.sentences[i]
            for event in lingo_sentence.events:
                print('## LINGO event')
                lingo_string = nlplingo_event_tostring(event)
                print(lingo_string)
                lingo_strings.append(lingo_string)
            print('')

            if '\n'.join(sorted(serif_strings)) != '\n'.join(sorted(lingo_strings)):
                strings1, strings2 = get_diff(serif_strings, lingo_strings)
                diff_docids[doc.docid] = '#### SERIF\n' + '\n========\n'.join(strings1) + '\n#### LINGO\n' + '\n========\n'.join(strings2)

    print('#### {} docids with differences'.format(str(len(diff_docids))))
    for docid in sorted(diff_docids):
        print('******** ' + docid)
        print(diff_docids[docid])


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)

    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))
    
    if args.mode == 'compare_bp_with_lingodoc':
        compare_bp_with_lingodoc(params)
    elif args.mode == 'compare_bp_with_serifxml':
        compare_bp_with_serifxml(params)
    elif args.mode == 'compare_serifxml_with_lingodoc':
        compare_serifxml_with_lingodoc(params)
