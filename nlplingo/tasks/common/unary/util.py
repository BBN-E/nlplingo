def _window_indices(target_indices, window_size):
    """
    +1
    Generates a window of indices around target_indices (token index within the sentence).
    window_size determines the number of neighbors recorded (in both directions) about target_indices,
    starting from 1 unit beyond the target index start/end.
    :type target_indices: nlplingo.common.utils.Struct
    :type window_size: int
    """
    indices = []
    indices.extend(range(target_indices.start - window_size, target_indices.start))
    indices.append(target_indices.head)
    indices.extend(range(target_indices.end + 1, target_indices.end + window_size + 1))
    return indices

def _calculate_position_data(token_indices, max_sent_length):
    """We capture positions of other words, relative to current word (corresponding to token_indices)

    tokens before token_indices will be assigned position relative to token_indices.start,
    tokens within the span of token_indices will be assigned a constant position of max_sentence_length,
    tokens after token_indices will be assigned position relative to token_indices.end

    Note that the code below adds max_sentence_length when assigning to pos_data.
    This is to avoid any negative or zero values in pos_data.

    :type token_indices: nlplingo.common.utils.Struct
    :type max_sentence_length: int
    """
    pos_data = []
    for i in range(max_sent_length):
        if i < token_indices.start:
            pos_data.append(i - token_indices.start + max_sent_length)
        elif token_indices.start <= i and i <= token_indices.end:
            pos_data.append(0 + max_sent_length)
        else:
            pos_data.append(i - token_indices.end + max_sent_length)
    return pos_data


def _calculate_position_data_variable(tokens, token_indices, max_sentence_length, padded=False):
    """
    if not padded (by default),
        tokens before token_indices will be assigned position relative to token_indices.start,
        tokens within the span of token_indices will be assigned a constant position of 0,
        tokens after token_indices will be assigned position relative to token_indices.end
    else if padded ...
        all tokens are assigned position relative to min(token_indices.start, max_sentence_length - 1)
        if the length of the sentence (length of tokens) is less than max_sentence_length, 0's are added till
        pos_data is of length max_sentence_length is filled

        it is possible that this mode could be further refactored so that
        'position relative to min(token_indices.start, max_sentence_length - 1)' can be altered

    :param tokens: list[nlplingo.text.text_span.Token]
    :param token_indices: nlplingo.common.utils.Struct
    :param max_sentence_length: int
    :param padded: boolean which controls behavior, described above
    """
    sentence_length = len(tokens)
    if not padded:
        start_idx = token_indices.start
        end_idx = token_indices.end
        pos_data = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
               list(range(1, sentence_length - end_idx))
    else:
        pos_data = []
        head_in_index = min(token_indices.start, max_sentence_length - 1)
        for i in range(min(sentence_length, max_sentence_length)):
            pos_data.append(i - head_in_index + max_sentence_length)

        while len(pos_data) < max_sentence_length:
            pos_data.append(0)

    return pos_data