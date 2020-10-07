from nlplingo.common.utils import IntPair, F1Score
from nlplingo.decoding.prediction_theory import DocumentPrediction, SentencePrediction, TriggerPrediction, \
    EventPrediction, ArgumentPrediction
from nlplingo.text.text_span import LabeledTextSpan, LabeledTextFrame


def get_frame_annotations_from_docs(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    :rtype: dict[str, list[list[nlplingo.text.text_span.LabeledTextFrame]]]
    """
    ret = dict()

    for doc in docs:
        doc_annotations = []
        for sentence in doc.sentences:
            sentence_annotations = []

            for event in sentence.events:
                anchor_spans = []
                for anchor in event.anchors:
                    span = LabeledTextSpan(IntPair(anchor.start_char_offset(), anchor.end_char_offset()), anchor.text, anchor.label)
                    span.start_token_index = anchor.tokens[0].index_in_sentence
                    span.end_token_index = anchor.tokens[-1].index_in_sentence
                    anchor_spans.append(span)

                argument_spans = []
                for argument in event.arguments:
                    span = LabeledTextSpan(IntPair(argument.start_char_offset(), argument.end_char_offset()), argument.text, argument.label)
                    span.start_token_index = argument.entity_mention.tokens[0].index_in_sentence
                    span.end_token_index = argument.entity_mention.tokens[-1].index_in_sentence
                    argument_spans.append(span)

                sentence_annotations.append(LabeledTextFrame(anchor_spans, argument_spans))

            doc_annotations.append(sentence_annotations)
        ret[doc.docid] = doc_annotations

    return ret

def get_trigger_annotations_from_docs(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    :rtype: dict[str, list[list[nlplingo.text.text_span.LabeledTextSpan]]]
    """
    ret = dict()

    for doc in docs:
        doc_annotations = []
        for sentence in doc.sentences:
            sentence_annotations = []

            for event in sentence.events:
                for anchor in event.anchors:
                    span = LabeledTextSpan(IntPair(anchor.start_char_offset(), anchor.end_char_offset()), anchor.text, anchor.label)
                    span.start_token_index = anchor.tokens[0].index_in_sentence
                    span.end_token_index = anchor.tokens[-1].index_in_sentence
                    sentence_annotations.append(span)

            doc_annotations.append(sentence_annotations)
        ret[doc.docid] = doc_annotations

    return ret

def get_matches_from_labeled_sequence(labels):
    """ From a list of BIO tags, condense into tuples of (start-token-index, end-token-index, tag)
    E.g.:
    index:  0  1     2       3         4         5     6   7
    token: In an explosion , many      people    will get hurt
    labels: O  O     O       B-Patient I-Patient O     O   O

    We generate: (3, 4, Patient)

    :type labels: list[str]
    """
    matches = []
    i = 0
    while i < len(labels):
        start = None
        end = None

        if labels[i] != 'O':
            if i == 0:
                start = i
            else:
                if labels[i].startswith('B-') or (labels[i] != labels[i - 1]):
                    start = i

        if start is not None:
            end = start + 1
            while end < len(labels):
                if labels[end] == 'O':
                    break
                elif labels[end].startswith('B-'):
                    break
                else:  # starts with 'I-'
                    if labels[start][2:] != labels[end][2:]:
                        break
                end += 1

        if start is not None and end is not None:
            matches.append((start, end - 1, labels[start][2:]))
            i = end
        else:
            i += 1

    return matches

def find_token_indices_of_markers(tokens, marker='$$$'):
    """
    :type tokens: list[str]
    """
    ret = []
    for i, token in enumerate(tokens):
        if token == marker:
            ret.append(i)
    return ret

def remove_tokens_at_indices(tokens, indices):
    """ param 'indices' is assumed to be sorted in increasing order
    :type tokens: list[object]
    :type indices: list[int]
    """
    for index in reversed(indices):
        tokens = tokens[0:index] + tokens[index+1:]
    return tokens

def convert_frames_to_prediction_theories(frames, docs):
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

            sentence_frames = doc_frames[i]  # get the list[LabeledTextFrame] for this sentence

            for frame in sentence_frames:
                # In a LabeledTextFrame, anchor_spans is a list. This allows the annotation data to specify multiple text spans as anchors.
                # However, in decoding, we always have just a single anchor_span. Hence we use frame.anchor_spans[0] below.
                # If there are multiple anchor_spans, then it is ambiguous which anchor_span will a given argument_span be associated with.
                anchor_span = frame.anchor_spans[0]
                #start_char = sentence.tokens[anchor_span.start_token_index].start_char_offset()
                #end_char = sentence.tokens[anchor_span.end_token_index].end_char_offset()

                start_char = sentence.tokens[anchor_span.start_token_index].start_char_offset()
                end_char = sentence.tokens[anchor_span.end_token_index].end_char_offset()

                anchor_span_text = sentence.get_text(start_char, end_char)
                if anchor_span_text is None:
                    anchor_span_text = anchor_span.text

                trigger_prediction = TriggerPrediction(start_char, end_char)
                trigger_prediction.text = anchor_span_text
                trigger_prediction.labels[anchor_span.label] = 1.0
                event_prediction = EventPrediction(trigger_prediction)

                for argument_span in frame.argument_spans:
                    #start_char = sentence.tokens[argument_span.start_token_index].start_char_offset()
                    #end_char = sentence.tokens[argument_span.end_token_index].end_char_offset()

                    start_char = sentence.tokens[argument_span.start_token_index].start_char_offset()
                    end_char = sentence.tokens[argument_span.end_token_index].end_char_offset()

                    argument_span_text = sentence.get_text(start_char, end_char)
                    if argument_span_text is None:
                        argument_span_text = argument_span.text

                    argument_prediction = ArgumentPrediction(start_char, end_char)
                    argument_prediction.text = argument_span_text
                    argument_prediction.labels[argument_span.label] = 1.0
                    event_prediction.arguments[str(len(event_prediction.arguments))] = argument_prediction

                sentence_prediction.events[str(len(sentence_prediction.events))] = event_prediction

            document_prediction.sentences[str(len(document_prediction.sentences))] = sentence_prediction

        document_predictions.append(document_prediction)

    return document_predictions


def print_scores(examples, predictions, sequence_types, score_outfile=None):
    """
    :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
    :type predictions: list[list[str]]
    :type score_outfile: str
    """
    assert len(examples) == len(predictions)

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        if len(example.words) != len(prediction):
            print('WARNING: len(example.words)=%d len(prediction)=%d' % (len(example.words), len(prediction)))
            assert len(example.words) > len(prediction)
            # to guard against overly long sentences where number of subwords exceed max_seq_len
            padding_length = len(example.words) - len(prediction)
            predictions[i] += ['O'] * padding_length
            # assert len(example.words) == len(prediction)

    y_true = []
    y_pred = []
    for example, prediction in zip(examples, predictions):
        y_true.extend(example.labels)
        y_pred.extend(prediction)

    # span_annotations = predictions_to_labeled_text_spans(examples, predictions, docs)
    # """:type: dict[str, list[list[LabeledTextSpan]]"""

    score_lines = []
    R = P = C = 0
    for class_label in sequence_types:
        if class_label != 'O':
            num_true = len([l for l in y_true if l == class_label])
            num_pred = len([l for l in y_pred if l == class_label])
            c = 0
            for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                if true == pred == class_label:
                    c += 1
            C += c
            R += num_true
            P += num_pred
            class_label_f1 = F1Score(c, num_true, num_pred, class_label)
            class_label_f1.calculate_score()
            #print(class_label_f1.to_string())
            score_lines.append(class_label_f1.to_string())

    overall_f1 = F1Score(C, R, P, 'overall')
    overall_f1.calculate_score()
    print(overall_f1.to_string())
    score_lines.append(overall_f1.to_string())

    if score_outfile is not None:
        with open(score_outfile, 'w', encoding='utf-8') as o:
            for line in score_lines:
                o.write(line)
                o.write('\n')

