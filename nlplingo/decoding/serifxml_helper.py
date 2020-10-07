import io
from collections import namedtuple

import serifxml3 as serifxml

"""
Almost all code here are rewritten from Java
"""


def get_anchor_node(span_start_offset, span_end_offset, parse_root):
    """
    Main purpose: mimic http://e-gitlab.bbn.com/text-group/jserif/blob/364-add-event-from-json/serif-util/src/main/java/com/bbn/serif/util/AddEventMentionFromJson.java#L254
    :param span_start_offset: int
    :param span_end_offset: int
    :param parse_root: serifxml.Parse
    :return: serifxml.SynNode
    """
    if parse_root is None:
        return None
    if parse_root.is_preterminal is True and parse_root.start_char == span_start_offset:
        return parse_root
    var3 = iter(parse_root.preterminals)
    child = None
    while True:
        child = next(var3, None)
        if child is None:
            raise RuntimeError("Something is wrong, we cannot find SynNode inside the span you indicated")
        if child.start_char >= span_start_offset and child.end_char <= span_end_offset:
            return get_anchor_node(span_start_offset, span_end_offset, child)


def find_proposition_by_node(anchor_node, sentence_theory):
    """
    http://e-gitlab.bbn.com/text-group/jserif/blob/master/serif/src/main/java/com/bbn/serif/theories/Propositions.java#L86
    :param anchor_node: serifxml.SynNode
    :param sentence_theory:serifxml.Sentence
    :return:serifxml.Proposition
    """
    var2 = iter(sentence_theory.proposition_set)
    while True:
        prop = next(var2, None)
        if prop is None:
            return None
        predHead = prop.head
        if predHead is None:
            continue
        if predHead is anchor_node:
            return prop
        for predHeadPreterm in predHead:
            if predHeadPreterm is anchor_node:
                return prop
    return None


def get_first_available_element_id(serifxml_document_theory):
    """
    This function exists because SerifXML requires each node has an id field. It's better to achieve it at serifxml.py,
    but we'll keep it here for now.
    :param serifxml_document_theory: serifxml.Document
    :return: str
    """
    keySet = set([int(float(i.replace('a', ''))) for i in serifxml_document_theory._idmap])
    for i in range(0, len(keySet) + 1):
        if i not in keySet:
            return "a{}".format(i)
    raise RuntimeError("It couldn't be true.")


def get_mention_with_fallback(span_start_offset, span_end_offset, mention_set, strict=False):
    """
    http://e-gitlab.bbn.com/text-group/jserif/blob/364-add-event-from-json/serif-util/src/main/java/com/bbn/serif/util/AddEventMentionFromJson.java#L316
    http://e-gitlab.bbn.com/text-group/jserif/blob/364-add-event-from-json/serif-util/src/main/java/com/bbn/serif/util/AddEventMentionFromJson.java#L328
    http://e-gitlab.bbn.com/text-group/jserif/blob/364-add-event-from-json/serif-util/src/main/java/com/bbn/serif/util/AddEventMentionFromJson.java#L340
    :param span_start_offset: int
    :param span_end_offset: int
    :param mention_set: serifxml.MentionSet
    :return:
    """
    for mention in mention_set:
        syn_node = mention.syn_node
        if syn_node.start_token.start_char == span_start_offset and syn_node.end_token.end_char == span_end_offset:
            return mention
    if strict is True:
        return None
    for mention in mention_set:
        syn_node = mention.syn_node
        if syn_node.start_token.start_char == span_start_offset or syn_node.end_token.end_char == span_end_offset:
            return mention
    for mention in mention_set:
        syn_node = mention.syn_node
        if syn_node.start_token.start_char >= span_start_offset and syn_node.end_token.end_char <= span_end_offset:
            return mention


def get_value_mention(span_start_offset, span_end_offset, value_mention_set):
    for value_mention in value_mention_set:
        if value_mention.start_token.start_char == span_start_offset and value_mention.end_token.end_char == span_end_offset:
            return value_mention
    return None


EventMentionArg_model = namedtuple('EventMentionArg', ['role', 'score', 'start_char', 'end_char'])


def create_event_mention_argument_identifier(event_mention_argument):
    """
    :param event_mention_argument: serifxml.EventMentionArg
    :return: EventMentionArg_model
    """
    if isinstance(event_mention_argument.value, serifxml.Mention):
        start_char = event_mention_argument.mention.syn_node.start_char
        end_char = event_mention_argument.mention.syn_node.end_char
    elif isinstance(event_mention_argument.value, serifxml.ValueMention):
        start_char = event_mention_argument.value_mention.start_char
        end_char = event_mention_argument.value_mention.end_char
    else:
        raise NotImplementedError("Please refer to serifxml.py's regarding section")
    return EventMentionArg_model(event_mention_argument.role, event_mention_argument.score, start_char, end_char)


def event_mention_args_eq(event_mention_1, event_mention_2):
    event_mention_arg_set1 = set()
    event_mention_arg_set2 = set()
    for idx, event_mention_argument in enumerate(event_mention_1.arguments):
        event_mention_arg_set1.add(create_event_mention_argument_identifier(event_mention_argument))
    for idx, event_mention_argument in enumerate(event_mention_2.arguments):
        event_mention_arg_set2.add(create_event_mention_argument_identifier(event_mention_argument))
    if len(event_mention_arg_set1.difference(event_mention_arg_set2)) > 0:
        return False
    if len(event_mention_arg_set2.difference(event_mention_arg_set1)) > 0:
        return False
    return True


EventMention_model = namedtuple('EventMention', ['event_type', 'score', 'start_char', 'end_char', 'arguments'])


def create_event_mention_identifier(event_mention, strict=False):
    """
    :param event_mention: serifxml.EventMention
    :return: EventMention_model
    """
    syn_node = event_mention.anchor_node
    event_mention_arg_set = set()
    if strict is True:
        for idx, event_mention_argument in enumerate(event_mention.arguments):
            event_mention_arg_set.add(create_event_mention_argument_identifier(event_mention_argument))
    return EventMention_model(event_mention.event_type, event_mention.score, syn_node.start_char, syn_node.end_char,
                              tuple(event_mention_arg_set))


def event_mention_eq(event_mention_1, event_mention_2, strict=False):
    if strict is False:
        return create_event_mention_identifier(event_mention_1) == create_event_mention_identifier(event_mention_2)
    else:
        return create_event_mention_identifier(event_mention_1) == create_event_mention_identifier(
            event_mention_2) and event_mention_args_eq(event_mention_1, event_mention_2)


def merge_event_mention_and_argument_in_serifxml(src_serifxml_str, dst_serifxml_str):
    """
    This function exists because when we have different trigger models, we need to decode document in a mapreduce fashion
    This is the reduce function
    :param src_serifxml: str
    :param dst_serifxml: str
    :return:
    """
    src_doc_theory = serifxml.Document(src_serifxml_str)
    dst_doc_theory = serifxml.Document(dst_serifxml_str)
    dst_sentence_theory_to_event_mention_id_map = dict()
    dst_event_mention_id_to_event_mention_map = dict()
    for sent_idx, dst_sentence_theory in enumerate(dst_doc_theory.sentences):
        for _, dst_event_mention in enumerate(dst_sentence_theory.event_mention_set):
            dst_event_mention_id = create_event_mention_identifier(dst_event_mention, True)
            dst_sentence_theory_to_event_mention_id_map.setdefault(sent_idx, set()).add(dst_event_mention_id)
            dst_event_mention_id_to_event_mention_map[dst_event_mention_id] = dst_event_mention
    for sent_idx, src_sentence_theory in enumerate(src_doc_theory.sentences):
        for _, src_event_mention in enumerate(src_sentence_theory.event_mention_set):
            src_event_mention_id = create_event_mention_identifier(src_event_mention, True)
            if src_event_mention_id in dst_sentence_theory_to_event_mention_id_map.get(sent_idx, set()):
                # EventMention and EventMentionArg both exist
                continue
            else:
                loose_event_mention_id_set_from_dst = set()
                loose_event_mention_id_to_event_mention_map = dict()
                dst_sentence_theory = dst_doc_theory.sentences[sent_idx]
                for _, event_mention_from_dst in enumerate(dst_doc_theory.sentences[sent_idx].event_mention_set):
                    dst_event_mention_id = create_event_mention_identifier(event_mention_from_dst, False)
                    loose_event_mention_id_set_from_dst.add(dst_event_mention_id)
                    loose_event_mention_id_to_event_mention_map[dst_event_mention_id] = event_mention_from_dst
                src_event_mention_id = create_event_mention_identifier(src_event_mention, False)
                if src_event_mention_id not in loose_event_mention_id_set_from_dst:
                    # Both EventMention and EventMentionArg not exist
                    # Create EventMention first
                    anchor_node = get_anchor_node(
                        src_event_mention_id.start_char,
                        src_event_mention_id.end_char,
                        dst_sentence_theory.parse.root)
                    anchor_prop = find_proposition_by_node(anchor_node, dst_sentence_theory)
                    new_id = get_first_available_element_id(dst_doc_theory)
                    new_event_mention = serifxml.EventMention(id=new_id,
                                                              anchor_node=anchor_node,
                                                              anchor_prop=anchor_prop,
                                                              score=src_event_mention.score,
                                                              event_type=src_event_mention.event_type,
                                                              genericity=src_event_mention.genericity,
                                                              polarity=src_event_mention.polarity,
                                                              tense=src_event_mention.tense,
                                                              modality=src_event_mention.modality,
                                                              owner=dst_sentence_theory)
                    dst_doc_theory.register_id(new_event_mention)
                    dst_event_mention = new_event_mention
                    dst_sentence_theory.event_mention_set._children.append(dst_event_mention)
                else:
                    # EventMention exist, partial EventMentionArg not exist
                    dst_event_mention = loose_event_mention_id_to_event_mention_map[src_event_mention_id]
                dst_exist_event_mention_arg_id_to_event_mention_arg_map = dict()
                for dst_event_mention_argument in dst_event_mention.arguments:
                    dst_event_mention_argument_id = create_event_mention_argument_identifier(dst_event_mention_argument)
                    dst_exist_event_mention_arg_id_to_event_mention_arg_map[
                        dst_event_mention_argument_id] = dst_event_mention_argument
                for src_event_mention_argument in src_event_mention.arguments:
                    src_event_mention_argument_id = create_event_mention_argument_identifier(src_event_mention_argument)
                    if src_event_mention_argument_id in dst_exist_event_mention_arg_id_to_event_mention_arg_map:
                        # Skip existing EventMentionArg
                        continue
                    else:
                        new_id = get_first_available_element_id(dst_doc_theory)
                        mention = get_mention_with_fallback(src_event_mention_argument_id.start_char,
                                                            src_event_mention_argument_id.end_char,
                                                            dst_sentence_theory.mention_set, True)
                        value_mention = get_value_mention(src_event_mention_argument_id.start_char,
                                                          src_event_mention_argument_id.end_char,
                                                          dst_sentence_theory.value_mention_set)
                        assert (isinstance(src_event_mention_argument.value,
                                           serifxml.Mention) and mention is not None) or (
                                       isinstance(src_event_mention_argument.value,
                                                  serifxml.ValueMention) and value_mention is not None)
                        if isinstance(src_event_mention_argument.value, serifxml.Mention):
                            new_event_mention_arg = serifxml.EventMentionArg(id=new_id, owner=dst_sentence_theory,
                                                                             role=src_event_mention_argument.role,
                                                                             mention=mention,
                                                                             score=src_event_mention_argument.score)
                        else:
                            new_event_mention_arg = serifxml.EventMentionArg(id=new_id, owner=dst_sentence_theory,
                                                                             role=src_event_mention_argument.role,
                                                                             value_mention=value_mention,
                                                                             score=src_event_mention_argument.score)
                        dst_doc_theory.register_id(new_event_mention_arg)
                        dst_event_mention.arguments.append(new_event_mention_arg)

    buffer = io.BytesIO()
    dst_doc_theory.save(buffer)
    buffer.seek(0)
    ret = buffer.read().decode('utf-8')
    return ret
