from nlplingo.tasks.common.binary.binary_event_entity import BinaryEventEntityFeatureGenerator


class EventArgumentFeatureGenerator(BinaryEventEntityFeatureGenerator):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        super(EventArgumentFeatureGenerator, self).__init__(extractor_params, hyper_params, feature_setting)

    def populate_example(self, example):
        """
        :type example: BinaryEventEntity
        """
        super(EventArgumentFeatureGenerator, self).populate_example(example)
        example.label[example.get_event_role_index()] = 1

    # TODO: We might want to wrap tokens and adj_mat in some sort of Background information class or something in future.
    """
    def generate_example(self, example, tokens, hyper_params, adj_mat=None):

        :type example: nlplingo.tasks.eventargument.EventArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type hyper_params: nlplingo.nn.extractor.HyperParameters


        anchor = example.anchor
        :type: nlplingo.text.text_span.Anchor
        argument = example.argument
        :type: nlplingo.text.text_span.EntityMention
        event_role = example.event_role

        # TODO need to handle multi-word arguments
        trigger_token_index = anchor.head().index_in_sentence
        # some triggers are multi-words, so we keep track of the start, end, and head token indices
        # trigger_token_indices = Struct(start=anchor.tokens[0].index_in_sentence,
        #                           end=anchor.tokens[-1].index_in_sentence, head=trigger_token_index)
        # we will just use the head token index, to be consistent with trigger training where we always use just the head for a trigger
        trigger_token_indices = Struct(start=trigger_token_index,
                                       end=trigger_token_index, head=trigger_token_index)

        arg_token_index = argument.head().index_in_sentence
        # some arguments are multi-words, so we keep track of the start, end, and head token indices
        arg_token_indices = Struct(start=argument.tokens[0].index_in_sentence,
                                   end=argument.tokens[-1].index_in_sentence, head=arg_token_index)

        # generate label data
        example.label[example.get_event_role_index()] = 1

        if self.features.trigger_argument_window:
            FeatureGenerator._assign_lexical_data(trigger_token_indices, arg_token_indices, tokens,
                                                               example, hyper_params.neighbor_distance)



        if self.features.argument_window:
            FeatureGenerator._assign_argument_lexical_data(arg_token_indices, tokens,
                                                               example, hyper_params.neighbor_distance)



        if self.features.event_embeddings:
            FeatureGenerator._assign_event_data(tokens, example)



        if self.features.sentence_word_embedding:
            FeatureGenerator.assign_sentence_data(tokens, example)



        if self.features.trigger_word_position and self.features.argument_word_position:
            FeatureGenerator._assign_position_data(trigger_token_indices, arg_token_indices, example,
                                                            hyper_params.max_sentence_length)



        if self.features.trigger_word_position_variable and self.features.argument_word_position_variable:
            FeatureGenerator._assign_position_data_variable(tokens, trigger_token_indices, arg_token_indices, example)

        if self.features.distance_between_trigger_argument:
            FeatureGenerator._assign_lexical_distance(trigger_token_indices, arg_token_indices, example)

        if self.features.sentence_ner_type:
            FeatureGenerator._assign_ner_data(tokens, example)

        if self.features.argument_ner_type:
            FeatureGenerator._assign_role_ner_data(example, arg_token_index)

        # assign 0, 1, 2, 3 depending on whether trigger and argument: overlap, has single comma in between, arg before trigger, or trigger before arg
        if self.features.trigger_argument_relative_position:
            FeatureGenerator._assign_relative_pos(example, trigger_token_indices, arg_token_indices)

        if self.features.argument_unique_ner_type_in_sentence:
            FeatureGenerator._assign_unique_role_type(example, arg_token_index)

        if self.features.argument_is_nearest_ner_type:
            FeatureGenerator._assign_nearest_type(example, trigger_token_index, arg_token_index)

        if self.features.argument_nominal:
            FeatureGenerator._assign_common_name_lex(example, tokens, arg_token_index)

        # directly use embeddings vector instead of indices
        if self.features.sentence_word_embedding_vector:
            FeatureGenerator.assign_sentence_vector_data(tokens, example)

        # directly use embeddings vector instead of indices, variable
        if self.features.sentence_word_embedding_vector_variable:
            FeatureGenerator.assign_sentence_vector_data_variable(tokens, example)



        if self.features.trigger_argument_window_vector:
            FeatureGenerator.assign_window_vector_data(trigger_token_indices, arg_token_indices, tokens, example,
                                                                        hyper_params.neighbor_distance)

        if self.features.argument_window_vector:
            FeatureGenerator.assign_argument_window_vector_data(arg_token_indices, tokens, example, hyper_params.neighbor_distance)

        TODO: fix graph models
        if self.features.adj_graph:
            FeatureGenerator._assign_adj_mat(tokens, example, adj_mat)

        if self.features.head_array:
            FeatureGenerator._assign_head_array(tokens, example, adj_mat[0])

        if self.features.dep_rels:
            FeatureGenerator._assign_dep_rels(tokens, example, adj_mat[1])
        """