from itertools import chain

import numpy as np

from nlplingo.common.utils import DEPREL_TO_ID
"""
This file is totally deprecated. It's kept here for legacy purposes (there may be some external projects that rely on this code).
"""

class FeatureGenerator(object):
    # window feature
    @staticmethod
    def _window_indices(target_indices, window_size):
        """
        +1
        Generates a window of indices around target_index (token index within the sentence)
        :type target_indices: nlplingo.common.utils.Struct
        :type window_size: int
        :type use_head: bool
        """
        indices = []
        indices.extend(range(target_indices.start - window_size, target_indices.start))
        indices.append(target_indices.head)
        indices.extend(range(target_indices.end + 1, target_indices.end + window_size + 1))
        return indices


    # window feature
    @classmethod
    def _get_token_windows(cls, tokens, window_size, arg1_token_indices, arg2_token_indices):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type window_size: int
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(chain(cls._window_indices(arg1_token_indices, window_size),
                                    cls._window_indices(arg2_token_indices, window_size))):
            if w < 0 or w >= len(tokens):
                continue
            ret.append((i, tokens[w]))
        return ret

    # window feature
    @classmethod
    def _get_token_window(cls, tokens, window_size, token_indices):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type window_size: int
        :type token_indices: nlplingo.common.utils.Struct
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        for i, w in enumerate(cls._window_indices(token_indices, window_size)):
            if w < 0 or w >= len(tokens):
                continue
            ret.append((i, tokens[w]))
        return ret



    # position feature
    @classmethod
    def _assign_lexical_distance(cls, arg1_token_indices, arg2_token_indices, example):
        """
        +1
        Captures the distance between eventtrigger and eventargument:
            0: they overlap
            1: eventtrigger and eventargument are neighboring tokens
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.util.Struct
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        index_list = [
            arg2_token_indices.start,
            arg1_token_indices.start,
            arg2_token_indices.end,
            arg1_token_indices.end
        ]
        result = np.argsort(index_list, kind='mergesort')

        if list(result[1:3]) == [2, 1] or list(result[1:3]) == [3, 0]:
            example.lex_dist = (index_list[result[2]] - index_list[result[1]])
        else:
            example.lex_dist = 0  # They overlap so there is zero distance

    # position feature
    @classmethod
    def _assign_relative_pos(cls, example, arg1_token_indices, arg2_token_indices):
        """
        +1
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        """

        def contains_comma(tokens, start, end):
            for tok in tokens[start:end]:
                if tok.text == ',':
                    return True
            return False

        ret_val = None
        index_list = [
            arg2_token_indices.start,
            arg1_token_indices.start,
            arg2_token_indices.end,
            arg1_token_indices.end
        ]
        result = np.argsort(index_list, kind='mergesort')

        # the intervals overlap
        # Since its a stable sort, if the first two elements after sorting are the arg_token_start and
        # trigger_token_start we can be assured that they are overlapping.

        if len({0, 1}.intersection(set(result[:2]))) == 2:
            ret_val = 0
        elif contains_comma(example.sentence.tokens, index_list[result[1]], index_list[result[2]]):
            ret_val = 1
        # arg2 is before arg1
        elif arg2_token_indices.end < arg1_token_indices.start:
            ret_val = 2
        # arg1 is before arg2
        elif arg1_token_indices.end < arg2_token_indices.start:
            ret_val = 3
        example.rel_pos = ret_val

    # eventargument/entity-mention feature
    @classmethod
    def _assign_common_name_lex(cls, example, tokens, arg_token_index):
        """
        +1
        Look within the eventargument's coreference chain, and assign a nominal (common noun) embedding
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type arg_token_index: int
        """
        if example.argument.is_common_noun() or example.argument.entity is None:
            example.common_word_vec = tokens[arg_token_index].vector_index
        else:
            offset_to_mention = dict()
            for mention in example.argument.entity.mentions:
                if mention.head() is not None and mention.is_common_noun():
                    offset_to_mention[mention.start_char_offset()] = mention
                    # ret_val = mention.head().vector_index

            if len(offset_to_mention) > 0:
                offset, mention = sorted(offset_to_mention.items(), key=lambda s: s[0])[0]
                example.common_word_vec = mention.head().vector_index
            else:
                example.common_word_vec = tokens[arg_token_index].vector_index

    # window feature
    @classmethod
    def _assign_lexical_data(cls, arg1_token_indices, arg2_token_indices, tokens, example, neighbor_dist):
        """
        +1
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type neighbor_dist: int
        :type use_argument_head: bool
        """
        # get the local token windows around the eventtrigger and eventargument
        token_windows = cls._get_token_windows(tokens, neighbor_dist, arg1_token_indices, arg2_token_indices)
        for (i, token) in token_windows:
            example.lex_data[i] = token.vector_index  # local window around arg1 and arg2

    # eventargument feature
    @classmethod
    def _assign_argument_lexical_data(cls, arg_token_indices, tokens, example, neighbor_dist):
        """
        +1
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type neighbor_dist: int
        :type use_argument_head: bool
        """
        # get the local token windows around the eventtrigger and eventargument
        token_windows = cls._get_token_window(tokens, neighbor_dist, arg_token_indices)
        for (i, token) in token_windows:
            example.argument_lex_data[i] = token.vector_index  # local window around eventargument

    # tasks specific feature
    @staticmethod
    def _assign_event_data(tokens, example):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            example.event_data[i] = example.event_domain.get_event_type_index(example.anchor.label)

    # eventargument/entity-mention feature
    @staticmethod
    def _assign_unique_role_type(example, arg_token_index):
        """
        +1
        checks whether the eventargument is the only entity-mention of that entity-type in the sentence
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type arg_token_index: int
        """
        ret_val = True
        arg_ne_type = example.ner_data[arg_token_index]
        for i, type in enumerate(example.ner_data):
            if i != arg_token_index and type == arg_ne_type:
                ret_val = False
                break
        example.is_unique_role_type = int(ret_val)

    # eventargument specific feature
    @staticmethod
    def _assign_nearest_type(example, arg1_token_index, arg2_token_index):
        """
        +1
        e.g. if the eventargument (arg2) is of type PER. Then check whether there is another PER entity-mention nearer (than current eventargument) to the arg1
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type trigger_token_index: int
        :type arg_token_index: int
        """
        ret_val = True
        target_dist = abs(arg1_token_index - arg2_token_index)
        arg_ne_type = example.ner_data[arg2_token_index]
        for i, type in enumerate(example.ner_data):
            if i != arg2_token_index and type == arg_ne_type:
                query_dist = abs(arg1_token_index - i)
                if query_dist < target_dist:
                    ret_val = False
                    break
        example.is_nearest_type = int(ret_val)

    # entity-mention feature
    @staticmethod
    def _assign_role_ner_data(example, arg_token_index):
        """
        +1
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :param arg_token_index: nlplingo.common.utils.Struct
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        example.role_ner[0] = example.event_domain.get_entity_type_index(token_ne_type[arg_token_index])

    # entity-mention feature
    @staticmethod
    def _assign_ner_data(tokens, example):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        assert len(token_ne_type) == len(tokens)
        for i, token in enumerate(tokens):
            example.ner_data[i] = example.event_domain.get_entity_type_index(token_ne_type[i])

    # NER with BIO of each token
    @staticmethod
    def _assign_entity_type_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
        """
        token_ne_bio_type = example.sentence.get_ne_type_with_bio_per_token()
        assert len(token_ne_bio_type) == len(tokens)
        for i, token in enumerate(tokens):
            example.entity_type_data[i] = example.event_domain.get_entity_bio_type_index(token_ne_bio_type[i])


    # sentence embeddings
    @staticmethod
    def assign_sentence_data(tokens, example):
        """
        +1
        Capture the embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            example.sentence_data[i] = token.vector_index

    # sentence embeddings
    @staticmethod
    def assign_sentence_vector_data(tokens, example):
        """
        +1
        Capture the actual word embeddings, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            if token.word_vector is not None:
                example.sentence_data_vector[i, :] = token.word_vector

    # sentence embeddings, variable length
    @staticmethod
    def assign_sentence_vector_data_variable(tokens, example):
        """
        +1
        Capture the actual word embeddings, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            if token.word_vector is not None:
                example.sentence_data_vector_variable.append(token.word_vector)
        example.sentence_data_vector_variable = np.vstack(example.sentence_data_vector_variable)

    # window embeddings
    @staticmethod
    def assign_window_vector_data(arg1_token_indices, arg2_token_indices, tokens, example, neighbor_dist):
        """We want to capture [word-on-left , target-word , word-on-right]
        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type neighbor_dist: int

        Returns: list[str]
        """
        token_windows = FeatureGenerator._get_token_windows(tokens, neighbor_dist, arg1_token_indices,
                                                                         arg2_token_indices)
        for (i, token) in token_windows:
            if token.word_vector is not None:
                example.window_data_vector[i, :] = token.word_vector

    # window embeddings
    @staticmethod
    def assign_argument_window_vector_data(arg_token_indices, tokens, example, neighbor_dist):
        """We want to capture [word-on-left , target-word , word-on-right]
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type neighbor_dist: int

        Returns: list[str]
        """
        token_windows = FeatureGenerator._get_token_window(tokens, neighbor_dist, arg_token_indices)
        for (i, token) in token_windows:
            if token.word_vector is not None:
                example.argument_window_vector[i, :] = token.word_vector

    # @staticmethod
    # def _assign_dep_data(example):
    #     """
    #     :type example: nlplingo.tasks.event_argument.EventArgumentExample
    #     """
    #     nearest_dep_child_distance = 99
    #     nearest_dep_child_token = None
    #     anchor_token_index = example.anchor.head().index_in_sentence
    #     for dep_r in example.anchor.head().child_dep_relations:
    #         if dep_r.dep_name == 'dobj':
    #             index = dep_r.child_token_index
    #             if abs(index - anchor_token_index) < nearest_dep_child_distance:
    #                 nearest_dep_child_distance = abs(index - anchor_token_index)
    #                 nearest_dep_child_token = example.sentence.tokens[dep_r.child_token_index]
    #
    #     if nearest_dep_child_token is not None:
    #         example.dep_data[0] = nearest_dep_child_token.vector_index
    #         example.anchor_obj = nearest_dep_child_token
    @staticmethod
    def _assign_dep_vector(example, tokens, trigger_index, argument_index):

        best_connect_dist = 10000
        for dep_rel in tokens[argument_index].dep_relations:
            if dep_rel.dep_direction == 'UP':

                modifier_text = dep_rel.dep_name
                target_text = tokens[dep_rel.connecting_token_index].text
                direction_modifier = 'I' if dep_rel.dep_direction == 'UP' else ''
                key = modifier_text + direction_modifier + '_' + target_text
                dep_data = tokens[argument_index].dep_rel_index_lookup.get(key, 0)

                if dep_data != 0:
                    connect_dist = abs(trigger_index - dep_rel.connecting_token_index)
                    if connect_dist < best_connect_dist:
                        example.arg_trigger_dep_data = dep_data
                        # This is important to the concept and wasnt there in initial testing JSF.
                        best_connect_dist = connect_dist

    @staticmethod
    def _assign_dep_data(example):
        """
        :type example: nlplingo.tasks.event_argument.EventArgumentExample
        """
        anchor_head_token = example.anchor.head()
        anchor_token_index = anchor_head_token.index_in_sentence

        candidate_tokens = set()
        nearest_distance = 99
        nearest_token = None
        for srl in example.sentence.srls:
            if srl.predicate_token == anchor_head_token:
                if 'A1' in srl.roles:
                    for text_span in srl.roles['A1']:
                        # index = text_span.tokens[0].index_in_sentence
                        candidate_tokens.add(text_span.tokens[0])
                        # if abs(index - anchor_token_index) < nearest_distance:
                        #    nearest_distance = abs(index - anchor_token_index)
                        #    nearest_token = example.sentence.tokens[index]

        if nearest_token is None:
            for dep_r in example.anchor.head().child_dep_relations:
                if dep_r.dep_name == 'dobj':
                    index = dep_r.child_token_index
                    candidate_tokens.add(example.sentence.tokens[index])
                    # if abs(index - anchor_token_index) < nearest_distance:
                    #    nearest_distance = abs(index - anchor_token_index)
                    #    nearest_token = example.sentence.tokens[index]

        candidate_tokens_filtered = [t for t in candidate_tokens if t.pos_category() != 'PROPN']

        final_candidates = []
        if len(candidate_tokens_filtered) > 0:
            final_candidates = candidate_tokens_filtered
        else:
            final_candidates = candidate_tokens

        for t in final_candidates:
            index = t.index_in_sentence
            if abs(index - anchor_token_index) < nearest_distance:
                nearest_distance = abs(index - anchor_token_index)
                nearest_token = example.sentence.tokens[index]

        if nearest_token is not None:
            example.dep_data[0] = nearest_token.vector_index
            example.anchor_obj = nearest_token

    # sentence texts
    @staticmethod
    def _assign_token_texts(tokens, max_sent_length):
        """
        +1
        Lexical text of each token in the sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        """
        token_texts = ['_'] * max_sent_length
        for i, token in enumerate(tokens):
            token_texts[i] = u'{0}'.format(token.text)  # TODO want to use token.vector_text instead?
        return token_texts

    @staticmethod
    def _calculate_position_data(token_indices, max_sent_length):
        """We capture positions of other words, relative to current word
        If the sentence is not padded with a None token at the front, then eg_index==token_index

        In that case, here is an example assuming max_sent_length==10 , and there are 4 tokens
        eg_index=0 , token_index=0    pos_data[0] = [ 0  1  2  3  4  5  6  7  8  9 ]  pos_index_data[0] = 0
        eg_index=1 , token_index=1    pos_data[1] = [-1  0  1  2  3  4  5  6  7  8 ]  pos_index_data[1] = 1
        eg_index=2 , token_index=2    pos_data[2] = [-2 -1  0  1  2  3  4  5  6  7 ]  pos_index_data[2] = 2
        eg_index=3 , token_index=3    pos_data[3] = [-3 -2 -1  0  1  2  3  4  5  6 ]  pos_index_data[3] = 3

        If the sentence is padded with a None token at the front, then eg_index==(token_index-1),
        and there are 5 tokens with tokens[0]==None

        eg_index=0 , token_index=1    pos_data[0] = [-1  0  1  2  3  4  5  6  7  8 ]  pos_index_data[0] = 1
        eg_index=1 , token_index=2    pos_data[1] = [-2 -1  0  1  2  3  4  5  6  7 ]  pos_index_data[1] = 2
        eg_index=2 , token_index=3    pos_data[2] = [-3 -2 -1  0  1  2  3  4  5  6 ]  pos_index_data[2] = 3
        eg_index=3 , token_index]4    pos_data[3] = [-4 -3 -2 -1  0  1  2  3  4  5 ]  pos_index_data[3] = 4

        * Finally, note that the code below adds self.gen.max_sent_length when assigning to pos_data.
        This is to avoid any negative values. For clarity of presentation, the above examples did not do this.

        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.tasks.eventtrigger.example.EventTriggerExample
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

    # position feature
    @staticmethod
    def _assign_position_data(arg1_token_indices, arg2_token_indices, example, max_sent_length):
        """
        +1
        NOTE: you do not know whether index_pair[0] refers to the trigger_token_index or arg_token_index.
        Likewise for index_pair[1]. You only know that index_pair[0] < index_pair[1]

        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type max_sent_length: int
        """
        # distance from eventtrigger
        example.trigger_pos_data[:] = FeatureGenerator._calculate_position_data(arg1_token_indices, max_sent_length)
        # distance from eventargument
        example.argument_pos_data[:] = FeatureGenerator._calculate_position_data(arg2_token_indices, max_sent_length)

    @staticmethod
    def _assign_adj_mat(tokens, example, edge_list):
        """
        +1
        Construct an adjacency matrix from an edge list.
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type edge_list: list[(node_index, node_index)]
        """
        for edge in edge_list:
            example.adj_mat[edge[0]][edge[1]] = 1
        example.adj_mat = example.adj_mat.flatten()

    @staticmethod
    def _assign_head_array(tokens, example, head_array):
        """
        +1
        Construct a graph in the form of a 'head array', 
        e.g. Node i has a parent with node index of heads[i] - 1
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type head_array: list[(node_index + 1)]
        """
        example.head_array = np.zeros(len(tokens), dtype='int32')
        for idx in range(len(tokens)):
            example.head_array[idx] = head_array[idx]

    @staticmethod
    def _assign_dep_rels(tokens, example, dep_rels):
        """
        +1
        Construct a dependency relation list.
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type edge_list: list[(node_index, node_index)]
        """
        example.dep_rels = np.zeros(len(tokens), dtype='int32')
        for idx in range(len(tokens)):
            example.dep_rels[idx] = dep_rels[idx]

    @staticmethod
    def get_positions(start_idx, end_idx, length):
        """ Get subj/obj position sequence. """
        return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
               list(range(1, length - end_idx))

    @staticmethod
    def _assign_position_data_variable(tokens, arg1_token_indices, arg2_token_indices, example):
        """
        +1
        NOTE: you do not know whether index_pair[0] refers to the trigger_token_index or arg_token_index.
        Likewise for index_pair[1]. You only know that index_pair[0] < index_pair[1]

        :type arg1_token_indices: nlplingo.common.utils.Struct
        :type arg2_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.eventargument.example.EventArgumentExample
        :type max_sent_length: int
        """
        # distance from eventtrigger
        sentence_length = len(tokens)
        example.trigger_pos_data = np.asarray(FeatureGenerator.get_positions(arg1_token_indices.start, arg1_token_indices.end, sentence_length))
        # distance from eventargument
        example.argument_pos_data = np.asarray(FeatureGenerator.get_positions(arg2_token_indices.start, arg2_token_indices.end, sentence_length))