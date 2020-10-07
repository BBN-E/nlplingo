from collections import defaultdict


def to_subwords(tokens, tokenizer):
    """ Tokenize the list of tokens, into subwords

    :type tokens: list[str]
    :type tokenizer: AutoTokenizer
    :rtype: list[SubWord]
    """
    unicodes = ['\u202b']
    subwords = []
    """:type list[SubWord]"""

    for i, token in enumerate(tokens):
        for j, subword in enumerate(tokenizer.tokenize(token)):
            for unicode in unicodes:
                if unicode in subword:
                    subword = subword.replace(unicode, '')

            if j == 0:  # the transformers codebase seem to always put ord 9601 at the beginning of each token
                assert ord(subword[0:1]) == 9601

            subwords.append(SubWord(len(subwords), subword, i, j == 0))

    return subwords


class SubWord(object):
    def __init__(self, subword_id, text, token_index, is_first_subword_in_token=None):
        """
        :type text: str
        :type token_index: int                  # the index of token that generated this subword
        :type is_first_subword_in_token: bool   # is this the first subword in the token
        """
        self.subword_id = subword_id    # this could simply be the subword index
        self.text = text
        self.token_index = token_index
        if is_first_subword_in_token is not None:
            self.is_first_subword_in_token = is_first_subword_in_token
        else:
            self.is_first_subword_in_token = self.is_first_subword_in_token(text)

    def is_first_subword_in_token(self, text):
        return ord(text[0:1]) == 9601


class SubWordPrediction(object):
    def __init__(self, text, label, score):
        self.text = text
        self.label = label
        self.score = score
        self.is_first_subword_in_token = self.is_first_subword_in_token(text)
        self.aligned_subwords = []
        """:type: list[SubWord]"""

    def is_first_subword_in_token(self, text):
        return ord(text[0:1]) == 9601

    def to_string(self):
        aligned_indices = ','.join(str(w.token_index) for w in self.aligned_subwords)
        return '{} {} {}'.format(self.text, self.label, aligned_indices)

    def to_string_highlight_multiple(self):
        aligned_indices = ','.join(str(w.token_index) for w in self.aligned_subwords)
        if ',' in aligned_indices:
            return 'MULTIPLE: {} {} {}'.format(self.text, self.label, aligned_indices)
        else:
            return '{} {} {}'.format(self.text, self.label, aligned_indices)


class AlignSubwordPredictionsToSubwordTokens(object):
    # we will ignore any predictions on the following texts, which doesn't really hold any semantic meaning
    words_to_skip = {'"', ':', ';', '``', "''", ',', '.', '?', '!', '(', ')', '[', ']', '{', '}', '<', '>', '`', "'",
                     '~', '_', '@', '#', '^', '*', '+', '=', '|', '/', '\\'}

    @staticmethod
    def _accept_prediction(p):
        """ Whether to accept a subword prediction, based on its text
        :type p: SubWordPrediction
        """
        if len(p.text) == 1 and ord(p.text) == 9601:
            return False

        if ord(p.text[0:1]) == 9601:
            t = p.text[1:]
            if t in AlignSubwordPredictionsToSubwordTokens.words_to_skip:
                return False

        return True

    @staticmethod
    def _get_start_end_based_on_first_subword_in_token(i, predictions, tokens):
        """
        :type predictions: list[SubWordPrediction]
        :type tokens: list[str]
        """
        start_token_index = 0
        for p in reversed(predictions[0:i]):
            if p.is_first_subword_in_token and len(p.aligned_subwords) == 1:
                start_token_index = p.aligned_subwords[0].token_index
                break
        end_token_index = len(tokens)
        for p in predictions[i + 1:]:
            if p.is_first_subword_in_token and len(p.aligned_subwords) == 1:
                end_token_index = p.aligned_subwords[0].token_index
                break
        return start_token_index, end_token_index

    @staticmethod
    def _get_start_end_based_on_any_subword_in_token(i, predictions, tokens):
        """
        :type predictions: list[SubWordPrediction]
        :type tokens: list[str]
        """
        start_token_index = 0
        for p in reversed(predictions[0:i]):
            if len(p.aligned_subwords) == 1:
                start_token_index = p.aligned_subwords[0].token_index
                break
        end_token_index = len(tokens)
        for p in predictions[i + 1:]:
            if len(p.aligned_subwords) == 1:
                end_token_index = p.aligned_subwords[0].token_index
                break
        return start_token_index, end_token_index

    @staticmethod
    def _filter_using_subword_texts(predictions):
        """
        :type predictions: list[SubWordPrediction]
        :rtype: list[SubWordPrediction]
        """
        return [p for p in predictions if AlignSubwordPredictionsToSubwordTokens._accept_prediction(p)]

    @staticmethod
    def _align_with_all_possible_subwords(predictions, subwords):
        """
        :type predictions: list[SubWordPrediction]
        :type subwords: list[SubWord]
        :rtype: list[SubWordPrediction]
        """
        for p in predictions:
            for w in subwords:
                if p.text == w.text:
                    p.aligned_subwords.append(w)

    @staticmethod
    def _constrain_predictions_based_on_surrounding_first_subwords(predictions, tokens):
        """
        indices:   1        2       3    4    5     6        7       8
        subwords: _The _alternative s _are _not _great _alternative  s
        predictions: ['_The':1] ['_alternative':2,7] ['s':3] ['_are':4]
        In the above, the prediction ['_alternative':2,7] which is based on subword that begins a token,
        has 2 plausible subword matches: [2,7]
        But the prediction (based on beginning subword) before & after it, matches indices [1] and [4] respectively.
        So that makes match [7] impossible. We thus filter this out.

        :type predictions: list[SubWordPrediction]
        :type tokens: list[str]
        """

        for i, prediction in enumerate(predictions):
            if prediction.is_first_subword_in_token and len(prediction.aligned_subwords) > 1:
                start_token_index, end_token_index = AlignSubwordPredictionsToSubwordTokens._get_start_end_based_on_first_subword_in_token(
                    i, predictions, tokens)
                new_aligned_subwords = []
                for w in prediction.aligned_subwords:
                    # TODO I should be able to do: start_token_index < w.token_index
                    if start_token_index < w.token_index < end_token_index:
                        new_aligned_subwords.append(w)
                prediction.aligned_subwords = new_aligned_subwords

    @staticmethod
    def _constrain_predictions_based_on_surrounding_subwords(predictions, tokens):
        for i, prediction in enumerate(predictions):
            if (not prediction.is_first_subword_in_token) and len(prediction.aligned_subwords) > 1:
                start_token_index, end_token_index = AlignSubwordPredictionsToSubwordTokens._get_start_end_based_on_any_subword_in_token(
                    i, predictions, tokens)
                new_aligned_subwords = []
                for w in prediction.aligned_subwords:
                    if start_token_index <= w.token_index and w.token_index <= end_token_index:
                        new_aligned_subwords.append(w)
                prediction.aligned_subwords = new_aligned_subwords

    @staticmethod
    def _prefer_arguments_to_be_nearer_trigger(predictions, tokens, ref_token_index):
        """ heuristically prefer arguments to be nearer trigger token
        :type predictions: list[SubWordPrediction]
        :type tokens: list[str]
        :type ref_token_index: int
        """
        for i, prediction in enumerate(predictions):
            if prediction.is_first_subword_in_token and len(prediction.aligned_subwords) > 1:
                nearest_distance_to_ref_token = len(tokens)
                nearest_subword = None
                for w in prediction.aligned_subwords:
                    distance = abs(w.token_index - ref_token_index)
                    if distance < nearest_distance_to_ref_token:
                        nearest_distance_to_ref_token = distance
                        nearest_subword = w
                assert nearest_subword is not None
                prediction.aligned_subwords = [nearest_subword]

    @staticmethod
    def _follow_surrounding_alignments_exact(predictions):
        """
        If my token_index equals previous prediction's token_index, then use it (unless I am a first subword)
        Else if my token_index equals next prediction's token_index, then use it

        :type predictions: list[SubWordPrediction]
        """
        for i, prediction in enumerate(predictions):
            if len(prediction.aligned_subwords) > 1:
                selected_subword = None
                if i > 0 and len(predictions[i - 1].aligned_subwords) == 1:
                    target_index = predictions[i - 1].aligned_subwords[0].token_index
                    for w in prediction.aligned_subwords:
                        if not prediction.is_first_subword_in_token and w.token_index == target_index:
                            selected_subword = w
                            break
                        elif prediction.is_first_subword_in_token and w.token_index > target_index:
                            selected_subword = w
                            break
                if selected_subword is None and (i + 1) < len(predictions) and \
                                len(predictions[i + 1].aligned_subwords) == 1:
                    target_index = predictions[i + 1].aligned_subwords[0].token_index
                    for w in prediction.aligned_subwords:
                        if w.token_index == target_index:
                            selected_subword = w
                            break
                if selected_subword is not None:
                    prediction.aligned_subwords = [selected_subword]

    @staticmethod
    def _follow_surrounding_alignments_off_by_one(predictions):
        """
        If my token_index equals previous prediction's token_index+1, then use it
        Else if my token_index equals next prediction's token_index-1, then use it

        :type predictions: list[SubWordPrediction]
        """
        for i, prediction in enumerate(predictions):
            if len(prediction.aligned_subwords) > 1:
                selected_subword = None
                if i > 0 and len(predictions[i - 1].aligned_subwords) == 1:
                    target_index = predictions[i - 1].aligned_subwords[0].token_index
                    for w in prediction.aligned_subwords:
                        if w.token_index == (target_index + 1):
                            selected_subword = w
                            break
                if selected_subword is None and (i + 1) < len(predictions) and len(
                        predictions[i + 1].aligned_subwords) == 1:
                    target_index = predictions[i + 1].aligned_subwords[0].token_index
                    for w in prediction.aligned_subwords:
                        if w.token_index == (target_index - 1):
                            selected_subword = w
                            break
                if selected_subword is not None:
                    prediction.aligned_subwords = [selected_subword]

    @staticmethod
    def _reduce_multiple_alignments(predictions):
        """
        :type predictions: list[SubWordPrediction]
        """
        text_indices_to_predictions = defaultdict(list)
        for prediction in predictions:
            key = '{}_{}'.format(prediction.text, '_'.join(str(w.token_index) for w in prediction.aligned_subwords))
            text_indices_to_predictions[key].append(prediction)
        for k in text_indices_to_predictions:
            if len(text_indices_to_predictions[k]) > 1:
                for i, p in enumerate(text_indices_to_predictions[k]):
                    # if len(p.aligned_subwords) != len(text_indices_to_predictions[k]):
                    #     for temp_p in text_indices_to_predictions[k]:
                    #         print(p.to_string_highlight_multiple())
                    # assert len(p.aligned_subwords) == len(text_indices_to_predictions[k])
                    if i < len(p.aligned_subwords):
                        p.aligned_subwords = [p.aligned_subwords[i]]
                    else:
                        p.aligned_subwords = [p.aligned_subwords[-1]]
                        # for p in predictions:
                        #     print(p.to_string_highlight_multiple())
                        # print('========')

    @staticmethod
    def _remove_extra_include_tags(predictions):
        """
        :type predictions: list[SubWordPrediction]
        """
        for prediction in predictions:
            if len(prediction.aligned_subwords) > 1 and prediction.label.startswith('I-'):
                prediction.aligned_subwords = []

    @staticmethod
    def align(subword_predictions, tokens, tokenizer, ref_token_index=None):
        """
        :type predictions: list[dict]
        :type tokens: list[str]
        :type tokenizer: AutoTokenizer
        :rtype: list[SubWordPrediction]
        """
        subwords = to_subwords(tokens, tokenizer)

        #print('subwords:', ' '.join(w.text for w in subwords))
        #print(subword_predictions)

        predictions = []
        """:type: list[SubWordPrediction]"""
        for prediction in subword_predictions:
            p = SubWordPrediction(prediction['word'], prediction['entity'], prediction['score'])
            predictions.append(p)

        predictions = AlignSubwordPredictionsToSubwordTokens._filter_using_subword_texts(predictions)

        AlignSubwordPredictionsToSubwordTokens._align_with_all_possible_subwords(predictions, subwords)

        AlignSubwordPredictionsToSubwordTokens._constrain_predictions_based_on_surrounding_first_subwords(predictions, tokens)

        if ref_token_index is not None:
            AlignSubwordPredictionsToSubwordTokens._prefer_arguments_to_be_nearer_trigger(predictions, tokens, ref_token_index)

        AlignSubwordPredictionsToSubwordTokens._constrain_predictions_based_on_surrounding_subwords(predictions, tokens)

        AlignSubwordPredictionsToSubwordTokens._follow_surrounding_alignments_exact(predictions)

        if ref_token_index is not None:  # this means we are doing prediction for arguments
            AlignSubwordPredictionsToSubwordTokens._follow_surrounding_alignments_off_by_one(predictions)

        if ref_token_index is None:         # we are doing trigger prediction
            AlignSubwordPredictionsToSubwordTokens._reduce_multiple_alignments(predictions)

        if ref_token_index is None:         # do a final check and reject all trigger 'I-' tag that still has multiple token indices
            AlignSubwordPredictionsToSubwordTokens._remove_extra_include_tags(predictions)
        # for p in predictions:
        #     print(p.to_string_highlight_multiple())
        # print('========')

        return predictions, subwords


def align_subwords_to_tokens(subwords, tokens):
    """ Mapping subwords to tokens should be a many-to-one mapping, since many subwords form a single token
    'tokens' should be Serif tokens
    Will return a dict from subword_id to token_index

    :type subwords: list[nlplingo.tasks.sequence.run.SubWord]
    :type tokens: list[nlplingo.text.text_span.Token]
    :rtype: dict[str, int]
    """
    strings_to_ignore = {'$$$'}

    subword_ids = [subword.subword_id for subword in subwords]
    subword_texts = [subword.text for subword in subwords]
    token_texts = [token.text for token in tokens]

    # let's first check that the total lengths between subwords and tokens are the same
    subword_len = 0
    for text in subword_texts:
        if ord(text[0:1]) == 9601:
            t = text[1:]
        else:
            t = text
        if t not in strings_to_ignore:
            subword_len += len(t)

    token_len = 0
    for text in token_texts:
        token_len += len(text)

    if subword_len != token_len:
        print('WARNING! subword_len != token_len, subword_len={} token_len={}'.format(str(subword_len), str(token_len)))
        print('] ||| ['.join(subword_texts))
        print('] ||| ['.join(token_texts))

    assert subword_len == token_len

    alignments = [-1] * len(subword_ids)
    current_subword_len = 0
    current_token_len = 0
    current_token_index = 0
    for i, subword_text in enumerate(subword_texts):

        if ord(subword_text[0:1]) == 9601:
            s_text = subword_text[1:]
        else:
            s_text = subword_text

        if s_text in strings_to_ignore:
            continue

        if (current_subword_len + len(s_text)) < current_token_len + len(token_texts[current_token_index]):
            current_subword_len += len(s_text)
            alignments[i] = current_token_index
        elif (current_subword_len + len(s_text)) == current_token_len + len(token_texts[current_token_index]):
            current_subword_len += len(s_text)
            current_token_len += len(token_texts[current_token_index])
            alignments[i] = current_token_index
            current_token_index += 1
        elif (current_subword_len + len(s_text)) > current_token_len + len(token_texts[current_token_index]):
            # E.g. raw text = '27,' , Serif breaks it up into 2 tokens '27' and ',' but this is still a single subword
            current_subword_len += len(s_text)
            current_token_len += len(token_texts[current_token_index])
            alignments[i] = current_token_index
            current_token_index += 1

            while current_token_len < current_subword_len:
                current_token_len += len(token_texts[current_token_index])
                current_token_index += 1

    # print('#### align_subwords_to_tokens')
    # s = ''
    # for i, text in enumerate(subword_texts):
    #     s += ' {}:{}:{}'.format(str(i), text, str(alignments[i]))
    # print('subwords:{}'.format(s))
    #
    # s = ''
    # for i, text in enumerate(token_texts):
    #     s += ' {}:{}'.format(str(i), text)
    # print('tokens:{}'.format(s))

    d = dict()
    for i, token_index in enumerate(alignments):
        d[subwords[i].subword_id] = token_index

    return d
