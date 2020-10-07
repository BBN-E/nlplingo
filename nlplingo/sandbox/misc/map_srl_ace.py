from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
from collections import defaultdict

import nlplingo.common.io_utils as io_utils
from nlplingo.annotation.ingestion import read_doc_annotation
from nlplingo.sandbox.common.parameters import Parameters
from nlplingo.text.text_span import IntegerPair


def prepare_docs(filelists):
    # read IDT and ENote annotations
    docs = read_doc_annotation(io_utils.read_file_to_list(filelists))
    print('num# docs = %d' % (len(docs)))

    for doc in docs:
        for sent in doc.sentences:
            doc.annotate_sentence_with_entity_mentions(sent)
            doc.annotate_sentence_with_events(sent)

    number_anchors = 0l
    number_args = 0
    number_assigned_anchors = 0
    number_assigned_args = 0
    number_assigned_multiword_anchors = 0
    event_type_count = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            number_anchors += event.number_of_anchors()
            number_args += event.number_of_arguments()
            event_type_count[event.label] += 1
        for sent in doc.sentences:
            for event in sent.events:
                number_assigned_anchors += event.number_of_anchors()
                number_assigned_args += event.number_of_arguments()
                for anchor in event.anchors:
                    if len(anchor.tokens) > 1:
                        number_assigned_multiword_anchors += 1


    print('In %d documents, #anchors=%d #assigned_anchors=%d #assigned_multiword_anchors=%d, #args=%d #assigned_args=%d' % \
          (len(docs), number_anchors, number_assigned_anchors, number_assigned_multiword_anchors, number_args, number_assigned_args))
    #print('Event type counts:')
    #for et in sorted(event_type_count.keys()):
    #    print('#{}: {}'.format(et, event_type_count[et]))
    """
    for doc in docs:
        for tasks in doc.events:
            if len(tasks.arguments) > 1:
                print(str(tasks.anchors[0].start_char_offset())+" "+str(tasks.anchors[0].end_char_offset())+" "+" "+str(tasks.arguments[0].start_char_offset())+" "+str(tasks.arguments[1].end_char_offset()))
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.srl is not None:
                    for role in token.srl.roles:
                        for index1, index2 in token.srl.roles[role]:
                            print(str(index1)+" "+str(index2)+" "+str(sentence.tokens[index1].start_char_offset()))

                    #print("srl:"+token.srl.predicate_label+" ", end=" ")
    """
    return docs

def calculate_srl_coverage_of_event(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    """

    overall_statistics = defaultdict(int)
    for doc in docs:
        doc_statistics = defaultdict(int)

        for sentence in doc.sentences:
            # first, let's capture the sets of character (start, end) for this sentence's anchors and tasks arguments
            anchor_spans = set()
            event_argument_spans = set()
            for event in sentence.events:
                anchor_spans.add(event.anchors[0].int_pair)
                for arg in event.arguments:
                    event_argument_spans.add(arg.int_pair)
            doc_statistics['anchor_count'] += len(anchor_spans)
            doc_statistics['argument_count'] += len(event_argument_spans)

            # now let's capture the SRL predicate and eventargument spans
            predicate_spans = set()
            srl_argument_spans = set()
            for token in sentence.tokens:
                if token.srl is not None:
                    predicate_spans.add(token.int_pair)
                    for srl_role in token.srl.roles:
                        for token_start, token_end in token.srl.roles[srl_role]:
                            srl_arg_char_start = sentence.tokens[token_start].start_char_offset()
                            srl_arg_char_end = sentence.tokens[token_end].end_char_offset()
                            srl_argument_spans.add(IntegerPair(srl_arg_char_start, srl_arg_char_end))

            # calculate coverge on anchors
            for anchor_span in anchor_spans:
                for predicate_span in predicate_spans:
                    if anchor_span.has_overlapping_boundary(predicate_span):
                        doc_statistics['covered_anchor_count'] += 1
                        break

            # calculate coverage on tasks arguments
            for event_arg_span in event_argument_spans:
                for srl_arg_span in srl_argument_spans:
                    if event_arg_span.has_overlapping_boundary(srl_arg_span):
                        doc_statistics['covered_argument_count'] += 1
                        break

        print('DOC {}, covered_anchors={}/{} covered_arguments={}/{}'.format(doc.docid,
                                                                             doc_statistics['covered_anchor_count'],
                                                                             doc_statistics['anchor_count'],
                                                                             doc_statistics['covered_argument_count'],
                                                                             doc_statistics['argument_count']))
        for k,v in doc_statistics.items():
            overall_statistics[k] += v

    print('OVERALL, covered_anchors={}/{} covered_arguments={}/{}'.format(overall_statistics['covered_anchor_count'],
                                                                         overall_statistics['anchor_count'],
                                                                         overall_statistics['covered_argument_count'],
                                                                         overall_statistics['argument_count']))

def match_predicates_anchors(docs):
    event_anchor_start_offset = -1
    event_anchor_end_offset = -1
    total_anchors = 0
    covered_anchors = 0
    covered_arguments = 0
    arguments_with_anchor_covered = 0
    for doc in docs:
        doc_total_anchors = 0
        doc_covered_anchors = 0
        print(doc.docid)
        for event in doc.events:
            if len(event.anchors) > 0:
                doc_total_anchors += 1
                event_anchor_start_offset = event.anchors[0].start_char_offset()
                event_anchor_end_offset = event.anchors[0].end_char_offset()
                print("tasks anchor: " + event.anchors[0].text + " " + "srl predicate: ", end="")

            for sentence in doc.sentences:
                for token in sentence.tokens:
                    if token.srl is not None:
                        if token.start_char_offset() == event_anchor_start_offset and token.end_char_offset() == event_anchor_end_offset:
                            print(token.text)
                            covered_anchors += 1
                            doc_covered_anchors += 1

                            arguments_with_anchor_covered += len(event.arguments)

                            for role in token.srl.roles:
                                for index1, index2 in token.srl.roles[role]:
                                    srl_predicate_start_offset = sentence.tokens[index1].start_char_offset()
                                    srl_predicate_end_offset = sentence.tokens[index2].end_char_offset()

                                    arguments_list = event.get_arguments_in_span(srl_predicate_start_offset, srl_predicate_end_offset)
                                    print("tasks arguments: ", end="")
                                    for arg in arguments_list:
                                        event_arg_start_offset = arg.start_char_offset()
                                        event_arg_end_offset = arg.end_char_offset()

                                        print(arg.text)
                                        #arguments_with_anchor_covered += 1

                                        if srl_predicate_start_offset <= event_arg_start_offset and srl_predicate_end_offset >= event_arg_end_offset:
                                            print("srl arguments: ", "end")

                                            for i in range(index1, index2+1):
                                                print(sentence.tokens[i].text+" ", end="")
                                            covered_arguments += 1
                                        print("")
            print("")
        print("")
    print("#covered anchors: " + str(covered_anchors)+" #arguments with anchor covered: "+str(arguments_with_anchor_covered)+" #covered arguments: " + str(covered_arguments))


def print_event_statistics(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    stats = defaultdict(int)
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                et = event.label
                stats[et] += 1
                for arg in event.arguments:
                    if arg.label == 'CyberAttackType' and et == 'CyberAttack':
                        print('CyberAttackType {} [{}] {}'.format(event.id, arg.text, sent.text.encode('ascii', 'ignore')))
                for anchor in event.anchors:
                    stats['{}.{}'.format(et, 'anchor')] += 1
                    if et == 'CyberAttack':
                        print('CyberAttack <{}> {}'.format(anchor.text, sent.text.encode('ascii', 'ignore')))
                for arg in event.arguments:
                    role = '{}.{}'.format(et, arg.label)
                    stats[role] += 1
    for key in sorted(stats.keys()):
        print('{}\t{}'.format(key, str(stats[key])))


def generate_event_statistics(params):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    docs = prepare_docs(params['data']['train']['filelist'])

    anchor_types = defaultdict(int)
    anchor_lines = []
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                anchor = event.anchors[0]
                anchor_types[anchor.label] += 1
                anchor_lines.append('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(doc.docid, anchor.id, anchor.label, anchor.text, anchor.head().text, anchor.start_char_offset(), anchor.end_char_offset()))

    with codecs.open(params['output.event_type_count'], 'w', encoding='utf-8') as o:
        for et in sorted(anchor_types):
            o.write('{}\t{}\n'.format(et, anchor_types[et]))

    with codecs.open(params['output.anchor_info'], 'w', encoding='utf-8') as o:
        for l in anchor_lines:
            o.write(l)
            o.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    filelists = params.get_string('filelists')
    docs = prepare_docs(filelists)
    #print_event_statistics(docs)
    #match_predicates_anchors(docs)
    calculate_srl_coverage_of_event(docs)
