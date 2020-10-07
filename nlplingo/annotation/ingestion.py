import codecs
import os
import re
import json
from collections import defaultdict

from nlplingo.annotation.ace import AceAnnotation
from nlplingo.annotation.serif import to_lingo_doc
from nlplingo.text.text_theory import Document
from nlplingo.annotation.idt import process_idt_file
from nlplingo.annotation.enote import process_enote_file
from nlplingo.annotation.spannotator import process_span_file
#from nlplingo.annotation.stanford_corenlp import process_corenlp_file
#from nlplingo.annotation.srl_column import process_srl_file
from nlplingo.embeddings.word_embeddings import DocumentContextualEmbeddings
from nlplingo.common import io_utils
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InputFileType(object):
    def __init__(self):
        self.text_file = None
        self.idt_file = None
        self.enote_file = None
        self.acetext_file = None  # ACE text file, where we only care about texts not within xml-tags
        self.apf_file = None  # ACE xml file
        self.span_file = None  # similar to Spannotator format
        self.corenlp_file = None
        self.srl_file = None
        self.serif_file = None
        self.lingo_file = None
        self.embedding_file = None


def parse_filelist_line(line):
    """
    :return: InputFileType
    """
    input_file_type = InputFileType()
    for file in line.strip().split():
        if file.startswith('TEXT:'):
            input_file_type.text_file = file[len('TEXT:'):]
        elif file.startswith('IDT:'):
            input_file_type.idt_file = file[len('IDT:'):]
        elif file.startswith('ENOTE:'):
            input_file_type.enote_file = file[len('ENOTE:'):]
        elif file.startswith('ACETEXT:'):
            input_file_type.acetext_file = file[len('ACETEXT:'):]
        elif file.startswith('APF:'):
            input_file_type.apf_file = file[len('APF:'):]
        elif file.startswith('SPAN:'):
            input_file_type.span_file = file[len('SPAN:'):]
        elif file.startswith('CORENLP'):
            input_file_type.corenlp_file = file[len('CORENLP:'):]
        elif file.startswith('SRL'):
            input_file_type.srl_file = file[len('SRL:'):]
        elif file.startswith('SERIF'):
            input_file_type.serif_file = file[len('SERIF:'):]
        elif file.startswith('LINGO'):
            input_file_type.lingo_file = file[len('LINGO:'):]
        elif file.startswith('EMBEDDING'):
            input_file_type.embedding_file = file[len('EMBEDDING:'):]

    if input_file_type.text_file is None and input_file_type.acetext_file is None and input_file_type.serif_file is None and input_file_type.lingo_file is None:
        raise ValueError('TEXT, ACETEXT, SERIF, or LINGO must be present!')
    return input_file_type


def read_doc_annotation(filelists, params):
    """
    :type filelists: list[str]
    Returns:
        list[nlplingo.text.text_theory.Document]
    """
    docs = []
    """:type docs: list[text.text_theory.Document]"""
    for file_index, line in enumerate(filelists):
        input_file_type = parse_filelist_line(line)

        # TODO: we probably want to restrict having only text_file, serif_file, or acetext_file
        if input_file_type.text_file is not None:
            docid = os.path.basename(input_file_type.text_file)
            text_f = codecs.open(input_file_type.text_file, 'r', encoding='utf-8')
            all_text = text_f.read()
            text_f.close()
            doc = Document(docid, all_text.strip())

        if input_file_type.serif_file is not None:
            doc = to_lingo_doc(input_file_type.serif_file,
                               params.get('add_serif_entity_mentions', True),
                               params.get('add_serif_event_mentions', False),
                               params.get('add_serif_event_event_relation_mentions', False),
                               params.get('add_serif_entity_entity_relation_mentions', False),
                               params.get('add_serif_prop_adj', False),
                               params.get('add_serif_dep_graph', False),
                               params.get('allow_anchor_as_event_argument', False)
                               )

        if input_file_type.acetext_file is not None:
            docid = re.match(r'(.*).sgm', os.path.basename(input_file_type.acetext_file)).group(1)
            text_list = AceAnnotation.process_ace_textfile_to_list(input_file_type.acetext_file)
            # sometimes, e.g. for ACE, we need to keep the sentence strings separate. In ACE sgm files, it contains
            # things like '&amp;' which Spacy normalizes to a single character '&' and Spacy thus changed the original
            # character offsets. This is bad for keeping correspondences with the .apf file for character offsets.
            # So we let it affect 1 sentence or 1 paragraph, but not the rest of the document.

            # Some ACE text files have words that end with a dash e.g. 'I-'. This presents a problem. The annotation
            # file annotates 'I', but Spacy keeps it as a single token 'I-', and then I won't be able to find
            # Spacy tokens to back the Anchor or EntityMention. To prevent these from being dropped, we will replace
            # all '- ' with '  '.
            text_list = [s.replace(r'- ', '  ') for s in text_list]
            text_list = [s.replace(r' ~', '  ') for s in text_list]
            text_list = [s.replace(r'~ ', '  ') for s in text_list]
            text_list = [s.replace(r' -', '  ') for s in text_list]
            text_list = [s.replace(r'.-', '. ') for s in text_list] # e.g. 'U.S.-led' => 'U.S. led', else Spacy splits to 'U.S.-' and 'led'
            text_list = [s.replace(r'/', ' ') for s in text_list]

            doc = Document(docid, text=None, sentence_strings=text_list)

        if input_file_type.lingo_file is not None:
            with codecs.open(input_file_type.lingo_file, 'r', encoding='utf-8') as f:
                doc = Document.from_json(json.load(f))

        if input_file_type.idt_file is not None:
            doc = process_idt_file(doc, input_file_type.idt_file)  # adds entity mentions
        if input_file_type.enote_file is not None:
            doc = process_enote_file(doc, input_file_type.enote_file, auto_adjust=True)  # adds events
        if input_file_type.apf_file is not None:
            doc = AceAnnotation.process_ace_xmlfile(doc, input_file_type.apf_file)
        if input_file_type.span_file is not None:
            doc = process_span_file(doc, input_file_type.span_file)
        if input_file_type.corenlp_file is not None:
            doc = process_corenlp_file(doc, input_file_type.corenlp_file)
        if input_file_type.srl_file is not None:
            doc = process_srl_file(doc, input_file_type.srl_file)
        if input_file_type.embedding_file is not None:
            if os.path.exists(input_file_type.embedding_file):
                npz_data = np.load(input_file_type.embedding_file, allow_pickle=True)
                DocumentContextualEmbeddings.load_embeddings_into_doc(doc, npz_data, params.get('average_embeddings', False))
            else:
                logger.critical("Embedding npz file {} does not exist".format(input_file_type.embedding_file))


        if doc is not None:
            docs.append(doc)

        if (file_index % 20) == 0:
            print('Read {} input documents out of {}'.format(str(file_index+1), str(len(filelists))))
    return docs


def count_annotations_in_docs(documents):
    number_anchors = 0
    number_args = 0
    number_assigned_anchors = 0
    number_assigned_args = 0
    number_assigned_multiword_anchors = 0
    event_type_count = defaultdict(int)
    for doc in documents:
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
          (len(documents), number_anchors, number_assigned_anchors, number_assigned_multiword_anchors, number_args, number_assigned_args))

def populate_doc_sentences_with_embeddings_and_annotations(docs, params, word_embeddings):

    if 'override_event_argument_annotation' in params:
        override_event_argument_annotation(docs, params['override_event_argument_annotation'])

    # apply Spacy for sentence segmentation and tokenization, using Spacy tokens to back Anchor and EntityMention
    #spacy_en = spacy.load('en')
    for index, doc in enumerate(docs):
        # if len(doc.sentences) == 0:
        #     doc.annotate_sentences(word_embeddings, spacy_en,
        #                            fuzzy_match_entitymention=params.get('entitymention.fuzzy_token_backing', False),
        #                            fuzzy_match_anchor=params.get('anchor.fuzzy_token_backing', False))
        # else:
        #     # read_doc_annotation above has already done sentence splitting and tokenization to construct
        #     # Sentence objects, e.g. using output from CoreNLP
        doc.annotate_sentences(word_embeddings, model=None,
                                   fuzzy_match_entitymention=params.get('entitymention.fuzzy_token_backing', False),
                                   fuzzy_match_anchor=params.get('anchor.fuzzy_token_backing', False))

        if (index % 20) == 0:
            print('Prepared {} input documents out of {}'.format(str(index+1), str(len(docs))))

    # em_labels = set()
    # for doc in docs:
    #     for sentence in doc.sentences:
    #         for em in sentence.entity_mentions:
    #             em_labels.add(em.label)
    # for label in sorted(em_labels):
    #     print('EM-LABEL {}'.format(label))
    # print(len(em_labels))

    # print('In train_test.prepare_docs')
    # for doc in docs:
    #     for sent in doc.sentences:
    #         for event in sent.events:
    #             for arg in event.arguments:
    #                 if len(arg.entity_mention.tokens) > 1:
    #                     print('Multiword argument: {} {}'.format(arg.entity_mention.label, ' '.join(token.text for token in arg.entity_mention.tokens)))
    # exit(0)

    count_annotations_in_docs(docs)

    #print('Event type counts:')
    #for et in sorted(event_type_count.keys()):
    #    print('#{}: {}'.format(et, event_type_count[et]))


def prepare_docs(filelists, word_embeddings, params):
    """
    :rtype: list[nlplingo.text.text_theory.Document]
    """
    # read IDT and ENote annotations
    docs = read_doc_annotation(io_utils.read_file_to_list(filelists), params)
    print('num# docs = %d' % (len(docs)))
    populate_doc_sentences_with_embeddings_and_annotations(docs, params, word_embeddings)
    return docs


def override_event_argument_annotation(docs, override_file):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    :type override_file: str
    """

    doc_annotations = defaultdict(list)
    with codecs.open(override_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for data in datas:
        doc_annotations[data['docid']].append(data)

    for doc in docs:
        docid = doc.docid
        if docid in doc_annotations:
            annotations = doc_annotations[docid]

            unfound_annotation_count = 0
            for annotation in annotations:
                target_event_type = annotation['trigger']['event_type']
                target_start = annotation['argument']['start']
                target_end = annotation['argument']['end']
                target_role = annotation['argument']['role']    # NOTE: if target_role == 'None', we remove the argument

                found = False
                for event in doc.events:
                    event_type = event.label
                    for arg in event.arguments:
                        start = arg.start_char_offset()
                        end = arg.end_char_offset()

                        if target_event_type == event_type and target_start == start and target_end == end:
                            arg.label = target_role
                            found = True
                            break

                if not found:
                    unfound_annotation_count += 1

            print('ingestion.override_event_argument_annotation: docid={} {}/{} annotations not found'.format(docid, str(unfound_annotation_count), str(len(annotations))))

        # we now remove arguments which have been tagged as 'None'
        for event in doc.events:
            new_arguments = []
            for arg in event.arguments:
                if arg.label != 'None':
                    new_arguments.append(arg)
            event.arguments = new_arguments

    role_counts = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            for arg in event.arguments:
                role_counts[arg.label] += 1

    for role in sorted(role_counts):
        print('ingestion.override_event_argument_annotation: #{} = {}'.format(role, str(role_counts[role])))

