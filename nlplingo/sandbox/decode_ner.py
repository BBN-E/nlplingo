from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import os

import spacy
from nlplingo.common.utils import IntPair
from nlplingo.embeddings.word_embeddings import WordEmbedding
from nlplingo.tasks.event_domain import AceDomain
from nlplingo.tasks.event_domain import CyberDomain
from nlplingo.tasks.event_domain import EventDomain
from nlplingo.tasks.event_domain import PrecursorDomain
from nlplingo.sandbox.common.parameters import Parameters
from nlplingo.sandbox.ner import Decoder
from nlplingo.sandbox.ner import NerFeature
from nlplingo.sandbox.ner import decode_sentence
from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_theory import Document


# from nlplingo.model.event_cnn import CNNTriggerModel
# from nlplingo.model.event_cnn import MaxPoolEmbeddedRoleModel

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')

    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    # load word embeddings
    word_embeddings = WordEmbedding(params, params.get_string('embedding.embedding_file'),
                                    params.get_int('embedding.vocab_size'), params.get_int('embedding.vector_size'))

    ner_fea = NerFeature(params)
    ner_decoder = Decoder(params)

    spacy_en = spacy.load('en')

    # initialize a particular tasks domain, which stores info on the tasks types and tasks roles
    event_domain = None
    if params.get_string('domain') == 'cyber':
        event_domain = CyberDomain()
    elif params.get_string('domain') == 'ace':
        event_domain = AceDomain()
    elif params.get_string('domain') == 'precursor':
        event_domain = PrecursorDomain()
    elif params.get_string('domain') == 'cyberattack':
        event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')

    # read in the list of input files to decode on
    input_filepaths = []
    with codecs.open(params.get_string('input_filelist'), 'r', encoding='utf-8') as f:
        for line in f:
            input_filepaths.append(line.strip())

    for i, input_filepath in enumerate(input_filepaths):
        print('decoding {} of {} files'.format(i, len(input_filepaths)))

        docid = os.path.basename(input_filepath)
        with codecs.open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        spacy_doc = spacy_en(content)
        ner_predictions = []
        for sent in spacy_doc.sents:
            ner_predictions.extend(decode_sentence(ner_fea, ner_decoder, content, sent, offset=0, content_type=params.get_string('content_type')))

        # create a document based on text content, add NER predictions as EntityMentions, then apply Spacy to
        # perform sentence segmentation and tokenization, and use Spacy tokens to back the EntityMentions
        doc = Document(docid, content)
        for i, p in enumerate(ner_predictions):
            id = 'em-{}'.format(i)
            doc.add_entity_mention(EntityMention(id, IntPair(p['start'], p['end']), p['text'], p['label']))

        out_lines = []
        for em in doc.entity_mentions:
            d = dict()
            d['start'] = em.start_char_offset()
            d['end'] = em.end_char_offset()
            d['text'] = em.text
            d['label'] = em.label
            out_lines.append(d)

        with codecs.open('{}/{}'.format(params.get_string('output_dir'), docid), 'w', encoding='utf-8') as o:
            o.write(json.dumps(out_lines, indent=4, cls=ComplexEncoder, ensure_ascii=False))

