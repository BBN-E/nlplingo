import os
import json
from collections import defaultdict

from nlplingo.tasks.common.feature.generator import FeatureGenerator
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer


class SequenceFeatureGenerator(FeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting, tokenizer, event_domain):
        """
        :type labels: list[str]     # we will pass in this during training
        :type hyper_params: nlplingo.nn.hyperparameters.HyperParameters
        """
        super(SequenceFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)
        self.tokenizer = tokenizer

        model_dir = extractor_params['model_file']

        # if tokenizer is not None:
        #     self.tokenizer = tokenizer
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        #self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)

        model_type = extractor_params['model_type']
        # if 'model_type' in params:
        #     self.model_type = params['model_type']
        # else:
        #     self.model_type = self._read_model_type(os.path.join(model_dir, 'config.json'))  # e.g. xlm-roberta

        # if labels is not None:
        #     self.labels = labels
        # else:
        #     self.labels = self._read_labels(os.path.join(model_dir, 'config.json'))
        #     """:type: list[str]"""
        #self.label_map = {label: i for i, label in enumerate(self.labels)}  # TODO can this be replaced by domain.ontology? Are the labels in alphabetical sorted order?

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        self.max_seq_length = hyper_params.max_sentence_length

        """ Example values when running with xlm-roberta
        cls_token_at_end: False
        cls_token: <s>
        cls_token_segment_id: 0
        sep_token: </s>
        sep_token_extra: False
        pad_on_left: False
        pad_token: 1
        pad_token_segment_id: 0
        pad_token_label_id: -100
        sequence_a_segment_id: 0
        mask_padding_with_zero True
        """
        self.cls_token_at_end = bool(model_type in ['xlnet'])  # xlnet has a cls token at the end
        #self.cls_token = self.tokenizer.cls_token  # e.g. <s>
        self.cls_token_segment_id = 2 if model_type in ['xlnet'] else 0
        #self.sep_token = self.tokenizer.sep_token  # e.g. </s>
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        self.sep_token_extra = bool(model_type in ['roberta'])
        self.pad_on_left = bool(model_type in ['xlnet'])  # pad on the left for xlnet
        #self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.pad_token_segment_id = 4 if model_type in ['xlnet'] else 0

        self.sequence_a_segment_id = 0
        self.mask_padding_with_zero = True

        self.statistics = defaultdict(int)

        self.event_domain = event_domain

    # def _read_model_type(self, config_filepath):
    #     with open(config_filepath, 'r', encoding='utf-8') as f:
    #         datas = json.load(f)
    #     return datas['model_type']

    # def _read_labels(self, config_filepath):
    #     """
    #     :rtype: list[str]
    #     """
    #     with open(config_filepath, 'r', encoding='utf-8') as f:
    #         datas = json.load(f)
    #     labels = [k for k, v in sorted(datas['label2id'].items(), key=lambda item: item[1])]
    #     return labels

    # +1
    def generate_example(self, example, label_map, tokenizer):
        """ Generate feature values for the given example
        :type example: nlplingo.tasks.sequence.example.SequenceExample
        :type label_map: dict[str, int]
        """
        self.cls_token = tokenizer.cls_token  # e.g. <s>
        self.sep_token = tokenizer.sep_token  # e.g. </s>
        self.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        tokens = []
        label_ids = []
        subword_to_token_indices = []  # for each subword, which token index it was derived from. This should follow `tokens` and `label_ids`
        word_index = 0
        # ==== Start off by populating with words and corresponding labels
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = tokenizer.tokenize('_')

            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
            # keep track of subword to token index
            subword_to_token_indices.extend([word_index] * len(word_tokens))
            word_index += 1

        #self.statistics['sequence_length={}'.format(str(len(tokens)))] += 1

        # ==== Truncate if too long
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if self.sep_token_extra else 2
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
            subword_to_token_indices = subword_to_token_indices[: (self.max_seq_length - special_tokens_count)]

        #subword_to_token_indices += [self.pad_token_label_id] * (self.max_seq_length - len(subword_to_token_indices))

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        #seq_length = len(tokens)
        # ==== Add separator
        tokens += [self.sep_token]
        label_ids += [self.pad_token_label_id]
        subword_to_token_indices += [self.pad_token_label_id]

        # ==== If need to add additional separator
        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [self.sep_token]
            label_ids += [self.pad_token_label_id]
            subword_to_token_indices += [self.pad_token_label_id]
        # ==== We now populate segment_ids based on the num# tokens thus far
        segment_ids = [self.sequence_a_segment_id] * len(tokens)

        # ==== Insert cls_token at the end, or at the front
        if self.cls_token_at_end:
            tokens += [self.cls_token]
            label_ids += [self.pad_token_label_id]
            subword_to_token_indices += [self.pad_token_label_id]
            segment_ids += [self.cls_token_segment_id]
        else:
            tokens = [self.cls_token] + tokens
            label_ids = [self.pad_token_label_id] + label_ids
            subword_to_token_indices = [self.pad_token_label_id] + subword_to_token_indices
            segment_ids = [self.cls_token_segment_id] + segment_ids

        # ==== Convert the tokens thus far, to their IDs. We will not be using `tokens` from here on
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # ==== The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # We now have: (word) input_ids, input_mask, segment_ids, label_ids

        # ==== Zero-pad up to the sequence length.
        seq_length = len(input_ids)     # TODO WARNING this would be wrong if we were using any model that pads on left
        padding_length = self.max_seq_length - len(input_ids)
        if self.pad_on_left:
            input_ids = ([self.pad_token] * padding_length) + input_ids
            input_mask = ([0 if self.mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([self.pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
            subword_to_token_indices = ([self.pad_token_label_id] * padding_length) + subword_to_token_indices
            tokens = ([self.pad_token] * padding_length) + tokens
        else:
            input_ids += [self.pad_token] * padding_length
            input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
            segment_ids += [self.pad_token_segment_id] * padding_length
            label_ids += [self.pad_token_label_id] * padding_length
            subword_to_token_indices += [self.pad_token_label_id] * padding_length
            tokens += [self.pad_token] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(subword_to_token_indices) == self.max_seq_length
        assert len(tokens) == self.max_seq_length

        """ Enable this if you want to see an example of the: input_ids, input_mask, segment_ids, label_ids
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        E.g.:
        tokens: <s> ▁Shi ft ing ▁al liance s ▁in ▁Israel ' s ▁par li ament , ▁the ▁K ness et , ▁today ▁appeared ▁to ▁offer ▁Prime ▁Minister ▁Men ach em ▁Begin ' s ▁ru ling ▁L ikud ▁coa li tion ▁the ▁prospect ▁of ▁sound er ▁foot ing ▁and ▁the ▁opposition ▁less ▁opportunity ▁to ▁topp le ▁the ▁government ▁without ▁new ▁election s . </s>
        input_ids: 0 8294 2480 214 144 87587 7 23 8254 25 7 366 150 11533 4 70 341 7432 126 4 18925 118775 47 18645 56195 33744 1111 934 195 121427 25 7 3114 2069 339 78452 5798 150 1363 70 109736 111 45730 56 57616 214 136 70 177986 40715 54591 47 24866 133 70 27759 15490 3525 81843 7 5 2 ... 1
        input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 ... 0
        segment_ids: 0 0 ... 0
        label_ids: -100 11 -100 -100 24 -100 -100 24 24 -100 -100 24 -100 -100 -100 24 24 -100 -100 -100 24 24 24 7 24 24 24 -100 -100 24 -100 -100 24 -100 24 -100 24 -100 -100 24 24 24 24 -100 24 -100 24 24 24 24 24 24 3 -100 24 24 24 24 8 -100 ... -100
        """

        example.input_ids = input_ids
        example.input_mask = input_mask
        example.segment_ids = segment_ids
        example.label_ids = label_ids
        example.subword_to_token_indices = subword_to_token_indices
        example.tokens = tokens
        example.seq_length = seq_length

        self.statistics['sequence_length={}'.format(str(seq_length))] += 1

    def print_statistics(self):
        print('#### SequenceFeatureGenerator statistics')
        for k in sorted(self.statistics):
            print(k, self.statistics[k])

    # placeholder for now
    def populate_example(self, example):
        """
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        """ Generate feature values for the given example
        :type example: nlplingo.tasks.sequence.example.SequenceExample
        :type label_map: dict[str, int]
        """
        self.cls_token = self.tokenizer.cls_token  # e.g. <s>
        self.sep_token = self.tokenizer.sep_token  # e.g. </s>
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        label_map = self.event_domain.sequence_types

        tokens = []
        label_ids = []
        subword_to_token_indices = []  # for each subword, which token index it was derived from. This should follow `tokens` and `label_ids`
        word_index = 0
        # ==== Start off by populating with words and corresponding labels
        for word, label in zip(example.words, example.labels):
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = self.tokenizer.tokenize('_')

            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
            # keep track of subword to token index
            subword_to_token_indices.extend([word_index] * len(word_tokens))
            word_index += 1

        #self.statistics['sequence_length={}'.format(str(len(tokens)))] += 1

        # ==== Truncate if too long
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if self.sep_token_extra else 2
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
            subword_to_token_indices = subword_to_token_indices[: (self.max_seq_length - special_tokens_count)]

        #subword_to_token_indices += [self.pad_token_label_id] * (self.max_seq_length - len(subword_to_token_indices))

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        #seq_length = len(tokens)
        # ==== Add separator
        tokens += [self.sep_token]
        label_ids += [self.pad_token_label_id]
        subword_to_token_indices += [self.pad_token_label_id]

        # ==== If need to add additional separator
        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [self.sep_token]
            label_ids += [self.pad_token_label_id]
            subword_to_token_indices += [self.pad_token_label_id]
        # ==== We now populate segment_ids based on the num# tokens thus far
        segment_ids = [self.sequence_a_segment_id] * len(tokens)

        # ==== Insert cls_token at the end, or at the front
        if self.cls_token_at_end:
            tokens += [self.cls_token]
            label_ids += [self.pad_token_label_id]
            subword_to_token_indices += [self.pad_token_label_id]
            segment_ids += [self.cls_token_segment_id]
        else:
            tokens = [self.cls_token] + tokens
            label_ids = [self.pad_token_label_id] + label_ids
            subword_to_token_indices = [self.pad_token_label_id] + subword_to_token_indices
            segment_ids = [self.cls_token_segment_id] + segment_ids

        # ==== Convert the tokens thus far, to their IDs. We will not be using `tokens` from here on
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # ==== The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # We now have: (word) input_ids, input_mask, segment_ids, label_ids

        # ==== Zero-pad up to the sequence length.
        seq_length = len(input_ids)     # TODO WARNING this would be wrong if we were using any model that pads on left
        padding_length = self.max_seq_length - len(input_ids)
        if self.pad_on_left:
            input_ids = ([self.pad_token] * padding_length) + input_ids
            input_mask = ([0 if self.mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([self.pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
            subword_to_token_indices = ([self.pad_token_label_id] * padding_length) + subword_to_token_indices
            tokens = ([self.pad_token] * padding_length) + tokens
        else:
            input_ids += [self.pad_token] * padding_length
            input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
            segment_ids += [self.pad_token_segment_id] * padding_length
            label_ids += [self.pad_token_label_id] * padding_length
            subword_to_token_indices += [self.pad_token_label_id] * padding_length
            tokens += [self.pad_token] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(subword_to_token_indices) == self.max_seq_length
        assert len(tokens) == self.max_seq_length

        """ Enable this if you want to see an example of the: input_ids, input_mask, segment_ids, label_ids
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        E.g.:
        tokens: <s> ▁Shi ft ing ▁al liance s ▁in ▁Israel ' s ▁par li ament , ▁the ▁K ness et , ▁today ▁appeared ▁to ▁offer ▁Prime ▁Minister ▁Men ach em ▁Begin ' s ▁ru ling ▁L ikud ▁coa li tion ▁the ▁prospect ▁of ▁sound er ▁foot ing ▁and ▁the ▁opposition ▁less ▁opportunity ▁to ▁topp le ▁the ▁government ▁without ▁new ▁election s . </s>
        input_ids: 0 8294 2480 214 144 87587 7 23 8254 25 7 366 150 11533 4 70 341 7432 126 4 18925 118775 47 18645 56195 33744 1111 934 195 121427 25 7 3114 2069 339 78452 5798 150 1363 70 109736 111 45730 56 57616 214 136 70 177986 40715 54591 47 24866 133 70 27759 15490 3525 81843 7 5 2 ... 1
        input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 ... 0
        segment_ids: 0 0 ... 0
        label_ids: -100 11 -100 -100 24 -100 -100 24 24 -100 -100 24 -100 -100 -100 24 24 -100 -100 -100 24 24 24 7 24 24 24 -100 -100 24 -100 -100 24 -100 24 -100 24 -100 -100 24 24 24 24 -100 24 -100 24 24 24 24 24 24 3 -100 24 24 24 24 8 -100 ... -100
        """

        example.input_ids = input_ids
        example.input_mask = input_mask
        example.segment_ids = segment_ids
        example.label_ids = label_ids
        example.subword_to_token_indices = subword_to_token_indices
        example.tokens = tokens
        example.seq_length = seq_length

        self.statistics['sequence_length={}'.format(str(seq_length))] += 1

