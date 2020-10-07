import json

#from transformers import ALL_PRETRAINED_MODEL_ARCHIVE_MAP
#from transformers.modeling_auto import MODEL_MAPPING
from nlplingo.tasks.sequence.utils import get_trigger_annotations_from_docs, get_frame_annotations_from_docs, \
    get_matches_from_labeled_sequence, find_token_indices_of_markers, remove_tokens_at_indices, \
    convert_frames_to_prediction_theories, print_scores
from serif.io.bpjson.abstract_events import Corpus

from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.sequence.feature import SequenceFeatureGenerator
from nlplingo.tasks.sequence import bpjson
from nlplingo.tasks.sequence.generator import SequenceExampleGenerator, align_tokens
from nlplingo.text.text_span import LabeledTextSpan, LabeledTextFrame
from nlplingo.common.utils import IntPair, F1Score
from nlplingo.decoding.prediction_theory import DocumentPrediction, SentencePrediction, EventPrediction, \
    TriggerPrediction, ArgumentPrediction
from nlplingo.nn.sequence_model import SequenceModel
from nlplingo.common.serialize_disk import load_from_serialized

import logging

logger = logging.getLogger(__name__)

#ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)
#MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

#TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


# +1
def create_sequence_labels(labels):
    """ Transforms input 'labels' into BIO format
    """
    ret = []
    ret.extend('B-{}'.format(label) for label in sorted(labels) if label != 'None')
    ret.extend('I-{}'.format(label) for label in sorted(labels) if label != 'None')
    ret.append('O')
    return ret

# +1
def get_trigger_annotations_from_bp(bp_json_file):
    """Extract trigger annotations as LabeledTextSpan
    In preparing the extractions from the BP JSON file, each sentence will be treated as its own doc.
    The sentence's entry-id will be treated as its docid. This is to be consistent with the SerifXML files provided,
    where each SerifXML file only contains 1 single sentence from the BP JSON file.

    :rtype: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]
    """
    ret = dict()

    bp_corpus = Corpus(bp_json_file)
    for _, doc in bp_corpus.docs.items():
        for sentence in doc.sentences:
            sentence_spans = []
            for _, abstract_event in sentence.abstract_events.items():  # for each abstract event in sentence
                for anchor_span in abstract_event.anchors.spans:        # for each anchor span in event
                    text = anchor_span.string
                    label = '{}.{}'.format(abstract_event.helpful_harmful, abstract_event.material_verbal)
                    sentence_spans.append(LabeledTextSpan(IntPair(None, None), text, label))
            ret[sentence.entry_id] = [sentence_spans]
    return ret

# +1
def get_frame_annotations_from_bp(bp_json_file):
    """Extract trigger and argument annotations as LabeledTextFrame
    In preparing the extractions from the BP JSON file, each sentence will be treated as its own doc.
    The sentence's entry-id will be treated as its docid. This is to be consistent with the SerifXML files provided,
    where each SerifXML file only contains 1 single sentence from the BP JSON file.

    :rtype: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]
    """
    ret = dict()

    bp_corpus = Corpus(bp_json_file)
    for _, doc in bp_corpus.docs.items():
        for sentence in doc.sentences:
            sentence_spans = []

            for _, abstract_event in sentence.abstract_events.items():  # for each abstract event in sentence
                anchor_spans = []
                for anchor_span in abstract_event.anchors.spans:        # for each anchor span in event
                    text = anchor_span.string
                    label = '{}.{}'.format(abstract_event.helpful_harmful, abstract_event.material_verbal)
                    anchor_spans.append(LabeledTextSpan(IntPair(None, None), text, label))

                argument_spans = []
                for arg in abstract_event.agents:
                    for span in arg.spans:
                        argument_spans.append(LabeledTextSpan(IntPair(None, None), span.string, 'AGENT'))
                for arg in abstract_event.patients:
                    for span in arg.spans:
                        argument_spans.append(LabeledTextSpan(IntPair(None, None), span.string, 'PATIENT'))
                sentence_spans.append(LabeledTextFrame(anchor_spans, argument_spans))

            ret[sentence.entry_id] = [sentence_spans]
    return ret

# ENTRY POINT
# WARNING! deprecated and not kept up to date
def train_trigger_from_bp(params, extractor):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    print('################ train_trigger_from_bp ###############')
    bp_filepath = params['bp_file']

    labels = create_sequence_labels(extractor.domain.event_types.keys())

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    annotations = get_trigger_annotations_from_bp(bp_filepath)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""

    model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), None, labels)

    example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, model.tokenizer)

    examples = example_generator.generate_spans_for_training(train_docs, annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""
    for example in examples:
        feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)

    global_step, tr_loss = model.train(examples)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    model.save_model()


# ENTRY POINT
# WARNING! deprecated and not kept up to date
def train_argument_from_bp(params, extractor):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    print('################ train_argument_from_bp ###############')

    bp_filepath = params['bp_file']

    labels = create_sequence_labels(extractor.domain.event_roles.keys())

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    annotations = get_frame_annotations_from_bp(bp_filepath)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""

    model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), None, labels)

    example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, model.tokenizer)

    examples = example_generator.generate_frames_for_training(train_docs, annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    for example in examples:        # populate each example with features
        feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)

    global_step, tr_loss = model.train(examples)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    model.save_model()

# ENTRY POINT
def train_trigger_from_docs(params, extractor):
    """ We are going to train a sequence model from params['data']['train']['filelist'],
    so if that points to a list of SerifXML files, then we are training from SerifXML files.

    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    print('################ train_trigger_from_docs ###############')

    #labels = create_sequence_labels(extractor.domain.event_types.keys())

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    annotations = get_trigger_annotations_from_docs(train_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""

    #model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), extractor.hyper_parameters)
    model = extractor.extraction_model

    #example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    #feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, model.tokenizer)

    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    examples = example_generator.generate_spans_for_training(train_docs, annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    for example in examples:
        feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)

    example_generator.print_statistics()
    feature_generator.print_statistics()

    if 'dev' in params['data']:
        dev_docs = prepare_docs(params['data']['dev']['filelist'], dict(), params)
        dev_annotations = get_trigger_annotations_from_docs(dev_docs)
        dev_examples = example_generator.generate_spans_for_training(dev_docs, dev_annotations)
        for example in dev_examples:
            feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)
    else:
        dev_examples = None

    model.train(examples, dev_examples)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    model.save_model()

# ENTRY POINT
def train_argument_from_docs(params, extractor):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    """
    print('################ train_argument_from_docs ###############')

    #labels = create_sequence_labels(extractor.domain.event_roles.keys())

    train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
    """:type: list[nlplingo.text.text_theory.Document]"""

    annotations = get_frame_annotations_from_docs(train_docs)
    """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]"""

    #model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), None, labels)
    model = extractor.extraction_model

    #example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    #feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, model.tokenizer)

    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    examples = example_generator.generate_frames_for_training(train_docs, annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""

    for example in examples:
        feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)

    # for example in examples:        # populate each example with features
    #     feature_generator.generate_example(example)

    example_generator.print_statistics()
    feature_generator.print_statistics()

    if 'dev' in params['data']:
        dev_docs = prepare_docs(params['data']['dev']['filelist'], dict(), params)
        dev_annotations = get_frame_annotations_from_docs(dev_docs)
        dev_examples = example_generator.generate_frames_for_training(dev_docs, dev_annotations)
        for example in dev_examples:
            feature_generator.generate_example(example, model.event_domain.sequence_types, model.tokenizer)
    else:
        dev_examples = None

    model.train(examples, dev_examples)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    model.save_model()

# ENTRY POINT
def train_ner(params, extractor, serialize_list, k_partitions=None, partition_id=None):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    """

    #labels = create_sequence_labels(extractor.domain.entity_types.keys())

    model = extractor.extraction_model
    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], dict(), params)
        """:type: list[nlplingo.text.text_theory.Document]"""

        examples = example_generator.generate(train_docs)
        feature_generator.populate(examples)
    else:
        examples, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)

    # for example in examples:        # populate each example with features
    #     feature_generator.generate_example(example, extractor.domain.sequence_types, model.tokenizer)

    global_step, tr_loss = model.train(examples)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    model.save_model()

####################################################

# +1
def decode_spans(extractor, docs):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    :type labels: list[str]

    :rtype: (list[nlplingo.tasks.sequence.example.SequenceExample], list[list[str]])
    """
    #model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), None, labels)

    #decoder = Decoder(extractor.model_file)
    decoder = extractor.extraction_model

    #example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    #feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, decoder.tokenizer)

    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    examples = example_generator.generate(docs)
    feature_generator.populate(examples)

    predictions = decoder.predict_sequence(examples, extractor.domain)
    # for each example, the sequence of predicted BIO
    assert len(examples) == len(predictions)

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        if len(example.words) != len(prediction):
            print('WARNING: len(example.words)=%d len(prediction)=%d' % (len(example.words), len(prediction)))

            assert len(example.words) > len(prediction)
            # to guard against overly long sentences where number of subwords exceed max_seq_len
            padding_length = len(example.words) - len(prediction)
            predictions[i] += ['O'] * padding_length
        #assert len(example.words) == len(prediction)

    return examples, predictions

# +1
def decode_argument(extractor, docs, annotations):
    """
    :type params: dict
    :type extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    :type annotations: dict[str, list[list[LabeledTextSpan]]]
    :type labels: list[str]
    :rtype: (list[nlplingo.tasks.sequence.example.SequenceExample], list[list[str]], list[str])

    annotations:
    * {docid} to a list, len(list) == number of sentences in the doc
    * Each element in list is a list of LabeledTextSpan in that sentence

    Each predicted BIO sequence is based on a marked up words sequence like so: w0 w1 w2 $$$ w3 w4 $$$ w5 w6
    In the returned 'examples', each example.words will be like such a marked up sequence
    In the returned 'anchor_labels', each anchor_label provides the label for the marked up span: (w3, w4) in above eg
    """
    #model = SequenceModel(params, extractor.extractor_params, extractor.domain, dict(), None, labels)

    #decoder = Decoder(extractor.model_file)
    #example_generator = SequenceExampleGenerator(extractor.domain, params, extractor.extractor_params, None)
    #feature_generator = SequenceFeatureGenerator(extractor.extractor_params, labels, decoder.tokenizer)

    decoder = extractor.extraction_model
    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    examples, anchor_labels = example_generator.generate_frames_for_decoding(docs, annotations)
    # the words in each example looks like: ... $$$ ... $$$ ... , and each str in anchor_labels denotes the label for the marked up span: $$$ ... $$$
    for example in examples:
        feature_generator.generate_example(example, decoder.event_domain.sequence_types, decoder.tokenizer)

    predictions = decoder.predict_sequence(examples, extractor.domain)

    assert len(examples) == len(predictions) == len(anchor_labels)

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        if len(example.words) != len(prediction):
            print('WARNING: len(example.words)=%d len(prediction)=%d' % (len(example.words), len(prediction)))

            assert len(example.words) > len(prediction)
            # to guard against overly long sentences where number of subwords exceed max_seq_len
            padding_length = len(example.words) - len(prediction)
            predictions[i] += ['O'] * padding_length
        #assert len(example.words) == len(prediction)

    return examples, predictions, anchor_labels

# +1
def predictions_to_labeled_text_spans(examples, predictions, docs):
    """
    :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
    :type predictions: list[list[str]]
    :type docs: list[nlplingo.text.text_theory.Document]

    :rtype: dict[str, list[list[LabeledTextSpan]]
    """
    span_annotations = dict()
    for doc in docs:
        l = [[] for _ in range(len(doc.sentences))]
        span_annotations[doc.docid] = l
        assert len(l) == len(doc.sentences)

    for example, bio in zip(examples, predictions):
        assert len(example.words) == len(bio)

        matches = get_matches_from_labeled_sequence(bio)
        """:type: list[(int, int, str)]"""

        for start, end, label in matches:
            labeled_span = LabeledTextSpan(IntPair(None, None), ' '.join(example.words[start:end + 1]), label)
            labeled_span.start_token_index = start
            labeled_span.end_token_index = end
            span_annotations[example.docid][example.sentence_index].append(labeled_span)

    return span_annotations

# +1
def decode_trigger_argument(params, trigger_extractor, argument_extractor, docs):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]

    :rtype: dict[str, list[list[LabeledTextFrame]]]
    We are returning the set of trigger spans and argument spans we predicted, as collections of LabeledTextFrame:
       {docid} -> [ [list of LabeledTextFrame for sentence 0],
                    [list of LabeledTextFrame for sentence 1],
                    ...,
                    [list of LabeledTextFrame for sentence n] ]
    In  the above, (n + 1) == len(doc.sentences)
    """

    #trigger_labels = create_sequence_labels(trigger_extractor.domain.event_types.keys())
    """:type: list[str]"""

    decode_with_gold_triggers = argument_extractor.extractor_params.get('decode_with_gold_triggers', False)
    if decode_with_gold_triggers:
        annotations = get_trigger_annotations_from_docs(docs)
        """:type: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]"""
        trigger_examples = trigger_extractor.example_generator.generate_spans_for_training(docs, annotations)
        """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""
        trigger_predictions = [eg.labels for eg in trigger_examples]
    else:
        trigger_examples, trigger_predictions = decode_spans(trigger_extractor, docs)
        """:type: list[nlplingo.tasks.sequence.example.SequenceExample], list[list[str]]"""
        # for each SequenceExample, we have the predicted list[str], which is its sequence of BIO predicted labels

    assert len(trigger_examples) == len(trigger_predictions)

    span_annotations = predictions_to_labeled_text_spans(trigger_examples, trigger_predictions, docs)
    """:type: dict[str, list[list[LabeledTextSpan]]"""

    #argument_labels = create_sequence_labels(argument_extractor.domain.event_roles.keys())
    """:type: list[str]"""

    argument_examples, argument_predictions, anchor_labels = decode_argument(argument_extractor, docs, span_annotations)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample], list[list[str]], list[str]"""
    assert len(argument_examples) == len(argument_predictions) == len(anchor_labels)

    # convert the above to LabeledTextFrame
    frame_annotations = dict()
    for doc in docs:
        l = [[] for _ in range(len(doc.sentences))]
        frame_annotations[doc.docid] = l
        assert len(l) == len(doc.sentences)

    for example, argument_bio, anchor_label in zip(argument_examples, argument_predictions, anchor_labels):
        assert len(example.words) == len(argument_bio)  # because argument_bio was predicted on example.words

        # example.words is marked up, like so: w0 w1 w2 $$$ w3 w4 $$$$ w5 w6,
        # where '$$$' surrounds the anchor span
        # We find the indices of the markers. In the above example, we should get: [3, 6]
        marker_indices = find_token_indices_of_markers(example.words)
        assert len(marker_indices) == 2

        words_without_markers = remove_tokens_at_indices(example.words, marker_indices)

        anchor_start_token_index = marker_indices[0]
        anchor_end_token_index = marker_indices[1] - 2
        # Why is there a -2 in the above? Assume that you have the following marked up word sequence:
        # 0 1 2  3  4 5  6  7 8  (token indices of marked up sequence)
        # 0 1 2 $$$ 3 4 $$$ 5 6  (original token indices)
        # On the above:
        # * marker_indices[0] == 3
        # * marker_indices[1] == 6 (you need to minus by 2, to get 4, which is the original token index)

        # TODO we want the anchor_text to be raw text, not concatenate of tokens
        anchor_text = ' '.join(words_without_markers[anchor_start_token_index: anchor_end_token_index + 1])
        anchor_span = LabeledTextSpan(IntPair(None, None), anchor_text, anchor_label)
        anchor_span.start_token_index = anchor_start_token_index
        anchor_span.end_token_index = anchor_end_token_index

        argument_bio_without_markers = remove_tokens_at_indices(argument_bio, marker_indices)

        argument_spans = []
        for (start_token_index, end_token_index, argument_label) in get_matches_from_labeled_sequence(argument_bio_without_markers):
            # TODO we want the argument_text to be raw text, not concatenate of tokens
            argument_text = ' '.join(words_without_markers[start_token_index: end_token_index+1])
            argument_span = LabeledTextSpan(IntPair(None, None), argument_text, argument_label)
            argument_span.start_token_index = start_token_index
            argument_span.end_token_index = end_token_index
            argument_spans.append(argument_span)

        labeled_frame = LabeledTextFrame([anchor_span], argument_spans)
        frame_annotations[example.docid][example.sentence_index].append(labeled_frame)

    return frame_annotations

###### Use the following when you are decoding on raw tokens and want to map onto Serif tokens
def _convert_frames_to_prediction_theories(frames, docs):
    """ LabeledTextFrame is something currently uniquely used by Sequence modeling,
    to enable capturing arbitrary predicted text spans, instead of predicted objects.
    So let's convert to the prediction objects in nlplingo.decoding.prediction_theory that are more generically used within NLPLingo.

    :type frames: dict[str, list[list[LabeledTextFrame]]]
    :type docs: list[nlplingo.text.text_theory.Document]

    {docid} -> [ [list of LabeledTextFrame for sentence 0],
                 [list of LabeledTextFrame for sentence 1],
                 ...,
                 [list of LabeledTextFrame for sentence n] ]
    In  the above, (n + 1) == len(doc.sentences)
    """
    document_predictions = []  # we will store our predictions here
    """:type: list[nlplingo.decoding.prediction_theory.DocumentPrediction]"""

    for doc in docs:
        document_prediction = DocumentPrediction(doc.docid)

        assert doc.docid in frames
        doc_frames = frames[doc.docid]
        assert len(doc_frames) == len(doc.sentences)

        for i, sentence in enumerate(doc.sentences):
            # prepare SentencePrediction object
            sentence_prediction = SentencePrediction(sentence.start_char_offset(), sentence.end_char_offset())
            sentence_prediction.id = doc.docid
            sentence_prediction.text = sentence.text

            # the decoding was done on SPACE tokens, but the only character offset information exists on the SERIF tokens
            # so we need to do a mapping from SPACE tokens to SERIF tokens
            token_alignments = align_tokens(sentence.text.split(), [token.text for token in sentence.tokens])
            assert len(token_alignments) == len(sentence.text.split())

            sentence_frames = doc_frames[i]  # get the list[LabeledTextFrame] for this sentence

            for frame in sentence_frames:
                # In a LabeledTextFrame, anchor_spans is a list. This allows the annotation data to specify multiple text spans as anchors.
                # However, in decoding, we always have just a single anchor_span. Hence we use frame.anchor_spans[0] below.
                # If there are multiple anchor_spans, then it is ambiguous which anchor_span will a given argument_span be associated with.
                anchor_span = frame.anchor_spans[0]
                #start_char = sentence.tokens[anchor_span.start_token_index].start_char_offset()
                #end_char = sentence.tokens[anchor_span.end_token_index].end_char_offset()

                aligned_start_token_index = token_alignments[anchor_span.start_token_index][0] # anchor_span.start_token_index is indexing the SPACE tokens
                aligned_end_token_index = token_alignments[anchor_span.end_token_index][1]     # anchor_span.end_token_index is indexing the SPACE tokens
                start_char = sentence.tokens[aligned_start_token_index].start_char_offset()
                end_char = sentence.tokens[aligned_end_token_index].end_char_offset()

                trigger_prediction = TriggerPrediction(start_char, end_char)
                trigger_prediction.text = anchor_span.text
                trigger_prediction.labels[anchor_span.label] = 1.0
                event_prediction = EventPrediction(trigger_prediction)

                for argument_span in frame.argument_spans:
                    #start_char = sentence.tokens[argument_span.start_token_index].start_char_offset()
                    #end_char = sentence.tokens[argument_span.end_token_index].end_char_offset()

                    aligned_start_token_index = token_alignments[argument_span.start_token_index][0]   # argument_span.start_token_index is indexing the SPACE tokens
                    aligned_end_token_index = token_alignments[argument_span.end_token_index][1]       # argument_span.end_token_index is indexing the SPACE tokens
                    start_char = sentence.tokens[aligned_start_token_index].start_char_offset()
                    end_char = sentence.tokens[aligned_end_token_index].end_char_offset()

                    argument_prediction = ArgumentPrediction(start_char, end_char)
                    argument_prediction.text = argument_span.text
                    argument_prediction.labels[argument_span.label] = 1.0
                    event_prediction.arguments[str(len(event_prediction.arguments))] = argument_prediction

                sentence_prediction.events[str(len(sentence_prediction.events))] = event_prediction

            document_prediction.sentences[str(len(document_prediction.sentences))] = sentence_prediction

        document_predictions.append(document_prediction)

    return document_predictions

# ENTRY POINT
def decode_sequence_trigger_argument(params, trigger_extractor, argument_extractor, docs):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    trigger_tokenization_type = trigger_extractor.extractor_params.get('tokenization_type', None)
    argument_tokenization_type = argument_extractor.extractor_params.get('tokenization_type', None)
    if trigger_tokenization_type == argument_tokenization_type == 'SPACE':
        for doc in docs:
            sentences = []
            for sentence in doc.sentences:
                sentences.append(sentence.copy_to_space_tokenization())
            doc.sentences = sentences

    frames = decode_trigger_argument(params, trigger_extractor, argument_extractor, docs)
    document_predictions = convert_frames_to_prediction_theories(frames, docs)

    # serialize out document_predictions
    d = dict()
    d['trigger'] = dict()
    for doc in document_predictions:
        d['trigger'][doc.docid] = doc.to_json()
    with open(params['predictions_file'], 'w', encoding='utf-8') as fp:
        json.dump(d, fp, indent=4, sort_keys=True, ensure_ascii=False)

    if 'bp_file' in params and 'output_bp' in params:
        # And the following is specific to BP. We convert the document_predictions to BP format, and serialize out a BP JSON file
        # we only want the corpus_id from the BP JSON file TODO is there a better way to get the corpus_id than feeding in the entire BP JSON file?
        with open(params['bp_file'], encoding="utf-8") as json_file:
            bp_data = json.load(json_file)
        corpus_id = bp_data["corpus-id"]

        bp_json = bpjson.document_prediction_to_bp_json(document_predictions, corpus_id)
        with open(params['output_bp'], 'w', encoding="utf-8") as outfile:
            json.dump(bp_json, outfile, indent=2, ensure_ascii=False)

def decode_ner(params, extractor):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    docs = prepare_docs(params['data']['test']['filelist'], dict(), params)
    
    #labels = create_sequence_labels(extractor.domain.entity_types.keys())
    """:type: list[str]"""

    examples, predictions = decode_spans(extractor, docs)
    """:type: list[nlplingo.tasks.sequence.example.SequenceExample], list[list[str]]"""
    # for each SequenceExample, we have the predicted list[str], which is its sequence of BIO predicted labels
    assert len(examples) == len(predictions)

    span_annotations = predictions_to_labeled_text_spans(examples, predictions, docs)
    """:type: dict[str, list[list[LabeledTextSpan]]"""

    # TODO
    # I am just going to print this out now. Some work needs to be done to put this into a better format
    for doc in docs:
        assert doc.docid in span_annotations
        doc_annotations = span_annotations.get(doc.docid)
        assert len(doc_annotations) == len(doc.sentences)

        for i, sentence in enumerate(doc.sentences):
            print(sentence.text)
            print(' ||| '.join(span.to_string() for span in doc_annotations[i]))


def test_ner(params, extractor):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    docs = prepare_docs(params['data']['test']['filelist'], dict(), params)

    # labels = create_sequence_labels(extractor.domain.entity_types.keys())
    """:type: list[str]"""

    model = extractor.extraction_model
    example_generator = extractor.example_generator
    feature_generator = extractor.feature_generator

    """:type: list[nlplingo.tasks.sequence.example.SequenceExample]"""
    example_generator.decode_mode = False # a small hack, because the candidate generation procedure is the same as the one at training time
    examples = example_generator.generate(docs)
    feature_generator.populate(examples)

    predictions = model.predict_sequence(examples, extractor.domain)
    """list[list[str]]"""

    print_scores(examples, predictions, extractor.domain.sequence_types, params['test.score_file'])


