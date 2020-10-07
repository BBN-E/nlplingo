import json
import os

import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from transformers import AutoTokenizer, AutoModelForTokenClassification




# TODO this should really be integrated into sequence_model.SequenceModel
# class Decoder(object):
#     def __init__(self, model_dir):
#         """
#         :type model_dir: str        # directory storing model files
#         """
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#
#         self.model_type = self._read_model_type(os.path.join(model_dir, 'config.json')) # e.g. xlm-roberta
#         self.labels = self._read_labels(os.path.join(model_dir, 'config.json'))
#
#         self.pad_token_label_id = CrossEntropyLoss().ignore_index
#
#         # self.max_seq_length = 256
#         #
#         # self.cls_token_at_end = bool(self.model_type in ['xlnet'])    # xlnet has a cls token at the end
#         # self.cls_token = self.tokenizer.cls_token                # e.g. <s>
#         # self.cls_token_segment_id = 2 if self.model_type in ['xlnet'] else 0
#         # self.sep_token = self.tokenizer.sep_token                # e.g. </s>
#         # # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
#         # self.sep_token_extra = bool(self.model_type in ['roberta'])
#         # self.pad_on_left = bool(self.model_type in ['xlnet'])         # pad on the left for xlnet
#         # self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
#         # self.pad_token_segment_id = 4 if self.model_type in ['xlnet'] else 0


"""
def decode_event_transformer_using_gold_trigger(params, argument_extractor, docs):
    argument_model = argument_extractor.model_file

    argument_tokenizer = AutoTokenizer.from_pretrained(argument_model)  # usage: argument_tokenizer.tokenize(<str>)

    if torch.cuda.is_available():
        argument_pipeline = pipeline(model=argument_model, config=argument_model, task='ner',
                                     tokenizer=('xlm-roberta-base', {'cache_dir': argument_extractor.extractor_params['cache_dir']}), device=0)
    else:
        argument_pipeline = transformers.pipeline(model=argument_model, config=argument_model, task='ner',
                                                  tokenizer=('xlm-roberta-base', {'cache_dir': argument_extractor.extractor_params['cache_dir']}))

    document_predictions = []

    with open(params['bp_file'], encoding="utf-8") as json_file:
        bp_data = json.load(json_file)
    corpus_id = bp_data["corpus-id"]

    bio_generator = BIOGenerator()
    keep_unannotated_sentences = argument_extractor.extractor_params['keep_unannotated_sentences']
    tokenization_type = argument_extractor.extractor_params['tokenization_type']  # SPACE, SERIF
    bio_lines = bio_generator.generate_trigger_bio(params['bp_file'], keep_unannotated_sentences, tokenization_type, docs)
    id_to_bio_tokens = separate_bio_lines_to_text_label(bio_lines)
    # id_to_bio_tokens is a dict where for each entry_id, we have:
    # id_to_bio_tokens[entry_id]['text'] = list[str]       # text tokens
    # id_to_bio_tokens[entry_id]['label'] = list[str]      # corresponding BIO label for each token

    for doc in docs:
        entry_id = doc.docid      # entry_id

        assert len(doc.sentences) == 1
        sentence = doc.sentences[0]

        document_prediction = DocumentPrediction(entry_id)
        sentence_prediction = SentencePrediction(sentence.start_char_offset(), sentence.end_char_offset())
        sentence_prediction.id = entry_id
        sentence_prediction.text = sentence.text

        assert entry_id in id_to_bio_tokens
        trigger_annotated_sentences = labels_to_annotated_sentence_text(id_to_bio_tokens[entry_id]['label'], id_to_bio_tokens[entry_id]['text'])
        ### for debug follows
        assert len(id_to_bio_tokens[entry_id]['label']) == len(id_to_bio_tokens[entry_id]['text'])
        text_label_strings = []
        for text, label in zip(id_to_bio_tokens[entry_id]['text'], id_to_bio_tokens[entry_id]['label']):
            text_label_strings.append('{}:{}'.format(text, label))
        #print(' '.join(text_label_strings))
        #for sent in trigger_annotated_sentences:
        #    print("  =>", sent.sentence_text)

        # starting from BIO on raw tokens, I want to align each trigger span (based on raw tokens) to SERIF tokens, to get the start_char and end_char
        token_indices_alignment = align_tokens(id_to_bio_tokens[entry_id]['text'], [token.text for token in sentence.tokens])
        # token_indices_alignment: list[(start_token_index, end_token_index)] , which references token indices in sentence.tokens
        assert len(token_indices_alignment) == len(id_to_bio_tokens[entry_id]['text'])

        matches = get_matches_from_labeled_sequence(id_to_bio_tokens[entry_id]['label'])
        trigger_offsets = []
        # this is list[(start_char, end_char, label)], where start_char & end_char are offsets w.r.t sentence.tokens
        for (start_index, end_index, label) in matches:
            target_start_token_index = token_indices_alignment[start_index][0]
            target_end_token_index = token_indices_alignment[end_index][1]

            start_char = sentence.tokens[target_start_token_index].start_char_offset()
            end_char = sentence.tokens[target_end_token_index].end_char_offset()
            trigger_offsets.append((start_char, end_char, label))

        assert len(trigger_annotated_sentences) == len(trigger_offsets)
        for i, trigger_sent in enumerate(trigger_annotated_sentences):
            assert trigger_sent.label == trigger_offsets[i][2]

        for trigger_sent_index, trigger_sent in enumerate(trigger_annotated_sentences):
            trigger_sentence_text = trigger_sent.sentence_text.replace("\\", "")

            (start_char, end_char, trigger_label) = trigger_offsets[trigger_sent_index]
            trigger_prediction = TriggerPrediction(start_char, end_char)
            trigger_prediction.text = sentence.text[start_char - sentence.start_char_offset(): end_char - sentence.start_char_offset()]
            trigger_prediction.labels[trigger_label] = 1.0
            event_prediction = EventPrediction(trigger_prediction)

            predicted_arguments = argument_pipeline(trigger_sentence_text)

            # determine token indices of trigger and trigger markers
            trigger_token_index = None
            marker_start_token_index = -1
            marker_end_token_index = -1
            tokens = trigger_sentence_text.split()
            for i, t in enumerate(tokens):
                if t == '$$$':
                    trigger_token_index = i + 1
                    break
            marker_start_token_index = trigger_token_index - 1
            i = trigger_token_index + 1
            while i < len(tokens):
                if tokens[i] == '$$$':
                    marker_end_token_index = i
                    break
                i += 1
            assert marker_start_token_index != -1 and marker_end_token_index != -1

            arg_predictions_with_alignments, argument_subwords = AlignSubwordPredictionsToSubwordTokens.align(predicted_arguments,
                                                                             trigger_sentence_text.split(),
                                                                             argument_tokenizer, trigger_token_index)

            argument_subword_id_to_token_index = align_subwords_to_tokens(argument_subwords, sentence.tokens)
            argument_offsets = predictions_to_annotation_offsets(arg_predictions_with_alignments, sentence.tokens,
                                                                argument_subword_id_to_token_index)
            for (start_char, end_char, argument_label) in argument_offsets:
                argument_prediction = ArgumentPrediction(start_char, end_char)
                argument_prediction.text = sentence.text[start_char - sentence.start_char_offset() : end_char - sentence.start_char_offset()]
                argument_prediction.labels[argument_label] = 1.0
                #event_prediction = EventPrediction(trigger_prediction)
                event_prediction.arguments[str(len(event_prediction.arguments))] = argument_prediction

            #print('{} {} {}'.format(str(marker_start_token_index), str(marker_end_token_index), trigger_sentence_text))

            sentence_prediction.events[str(len(sentence_prediction.events))] = event_prediction

            # found_tokens = find_tokens_using_offsets(sentence.tokens, event_prediction.trigger.start, event_prediction.trigger.end)
            # print('EVENT {}:{}'.format(list(event_prediction.trigger.labels)[0], ' '.join(token.text for token in found_tokens)))

            # arg_string = ''
            # for argument in event_prediction.arguments.values():
            #     found_arg_tokens = find_tokens_using_offsets(sentence.tokens, argument.start, argument.end)
            #     arg_string += '{}:{} '.format(list(argument.labels)[0], ' '.join(token.text for token in found_arg_tokens))
            # print('    ARG {}'.format(arg_string))

        document_prediction.sentences[str(len(document_prediction.sentences))] = sentence_prediction
        document_predictions.append(document_prediction)

    bp_json = bpjson.document_prediction_to_bp_json(document_predictions, corpus_id)
    with open(params['output_bp'], 'w', encoding="utf-8") as outfile:
        json.dump(bp_json, outfile, indent=2, ensure_ascii=False)

    d = dict()
    d['trigger'] = dict()
    for doc in document_predictions:
        d['trigger'][doc.docid] = doc.to_json()

    with open(params['predictions_file'], 'w', encoding='utf-8') as fp:
        json.dump(d, fp, indent=4, sort_keys=True, ensure_ascii=False)
"""

