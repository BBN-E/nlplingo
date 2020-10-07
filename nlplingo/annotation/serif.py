
import sys

from nlplingo.text.dependency_relation import DependencyRelation

if sys.version_info[0] == 2:
  import serifxml as serifxml
  from serifxml import Document as serifDoc
elif sys.version_info[0] == 3:
  import serifxml3 as serifxml
  from serifxml3 import Document as serifDoc

from nlplingo.common.utils import IntPair
from nlplingo.text.text_theory import Document as lingoDoc
from nlplingo.text.text_span import Token
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_theory import Event
from nlplingo.text.text_theory import EventEventRelation
from nlplingo.text.text_theory import EntityRelation
from nlplingo.text.text_span import EventArgument
from nlplingo.text.text_span import Anchor

from nlplingo.common.utils import DEPREL_TO_ID, only1

def get_snippet(serif_doc,sentence_theory):
    sentence_start = sentence_theory.token_sequence[0].start_char
    sentence_end = sentence_theory.token_sequence[-1].end_char

    sentence_start_edt = sentence_theory.token_sequence[0].start_edt
    sentence_end_edt = sentence_theory.token_sequence[-1].end_edt

    #print('serif.get_snippet={}'.format(serif_doc.get_original_text_substring(sentence_start, sentence_end)))

    return serif_doc.get_original_text_substring(sentence_start, sentence_end), sentence_start_edt, sentence_end_edt


def to_tokens(st):
    """
    :type st: serifxml.SentenceTheory

    Returns: list[nlplingo.text.text_span.Token]
    """
    if st.token_sequence and st.pos_sequence:
        token_dict = dict()
        """:type: dict[str, nlplingo.text.text_span.Token]"""   # token.id -> Token
        tokens = []                     # we will return this
        child_id_to_head_id = dict()    # child token id -> head token id ; this is for dependency relation
        token_id_to_pos = dict()
        token_id_to_upos = dict()
        token_id_to_deprel = dict()

        for i, token in enumerate(st.token_sequence):
            tok = Token(IntPair(token.start_edt, token.end_edt + 1), i, token.text, token.lemma, pos_tag=None)
            tokens.append(tok)
            token_dict[token.id] = tok
            if token.head:              # will be False for root of dependency graph
                child_id_to_head_id[token.id] = token.head.id

        for pos in st.pos_sequence:
            if pos.token:               # pos.token is needed for us to know which token this annotation is for
                if pos.tag:
                    token_id_to_pos[pos.token.id] = pos.tag
                if pos.upos:
                    token_id_to_upos[pos.token.id] = pos.upos
                if pos.dep_rel:
                    token_id_to_deprel[pos.token.id] = pos.dep_rel

        for token_id, pos in token_id_to_pos.items():
            if token_id in token_dict:
                token_dict[token_id].pos_tag = pos

        for token_id, upos in token_id_to_upos.items():
            if token_id in token_dict:
                token_dict[token_id].pos_tag_alternate = upos

        for token_id, deprel in token_id_to_deprel.items():
            if token_id in token_dict:
                if token_id in child_id_to_head_id:
                    head_id = child_id_to_head_id[token_id]
                    assert head_id in token_dict
                    token_dict[token_id].add_dep_relation(
                        DependencyRelation(deprel, DependencyRelation.up(), token_dict[head_id].index_in_sentence + 1)) # we reserve 0 for the root
                else:   # this is the root of the dependency graph, so it is not the child of any other token
                    token_dict[token_id].add_dep_relation(DependencyRelation(deprel, DependencyRelation.up(), 0))

        for token in tokens:
            if len(token.dep_relations) != 1:
                print('WARNING: len(token.dep_relations)=', len(token.dep_relations), ' when it should be == 1')

        #for token in tokens:
        #    print('%d %d-%d %s %s %s %s %d %s' % (
        #    token.index_in_sentence, token.start_char_offset(), token.end_char_offset(), token.text, token.lemma,
        #    token.pos_tag, token.pos_tag_alternate, token.dep_relations[0].connecting_token_index,
        #    token.dep_relations[0].dep_name))

        count_root = 0
        for token in tokens:
            if len(token.dep_relations) > 0:
                if token.dep_relations[0].connecting_token_index == 0:
                    count_root += 1
        if count_root != 1:
            print('WARNING: count_root=', count_root)	# these are probably rightfully multiple sentences, that are somehow stuck together

        return tokens

    elif st.parse and st.parse.root:   # if there is no pos sequence, we will get the pos-tag from the parse tree
        tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""

        root = st.parse.root
        """:type: serifxml.SynNode"""
        for i, t in enumerate(root.terminals):
            t_text = t.text
            t_start = t.start_token.start_edt
            t_end = t.end_token.end_edt
            t_pos_tag = t.parent.tag
            # we do a +1 because this has been the assumption in nlplingo
            token = Token(IntPair(t_start, t_end + 1), i, t_text, lemma=None, pos_tag=t_pos_tag)
            # token.pos_tag_alternate = pos_tags[i]
            tokens.append(token)
        return tokens

    else:   # get tokens from st.token_sequence
        tokens = []

        for i, token in enumerate(st.token_sequence):
            tok = Token(IntPair(token.start_edt, token.end_edt + 1), i, token.text, token.lemma, pos_tag=None)
            tokens.append(tok)

        return tokens


def add_names(st, doc):
    """
    :type st: serifxml.SentenceTheory
    :type doc: nlplingo.text.text_theory.Document
    """
    if st.name_theory is not None:
        for m in st.name_theory:
            start = m.start_edt
            end = m.end_edt + 1
            #print('serif.add_names: start={} end={}'.format(start, end))
            m_exists = False
            for em in doc.entity_mentions:
                if em.start_char_offset() == start and em.end_char_offset() == end:
                    m_exists = True
                    break
            if not m_exists:
                em = EntityMention(m.id, IntPair(start, end), m.text, m.entity_type)
                doc.add_entity_mention(em)

def add_entity_mentions(st, s, doc):
    """
    :type st: serifxml.SentenceTheory
    :type s: nlplingo.text.text_span.Sentence
    :type doc: nlplingo.text.text_theory.Document
    """

    if st.mention_set is not None:
        for m in st.mention_set:
            if m.entity_subtype != 'UNDET':
                m_type = '{}.{}'.format(m.entity_type, m.entity_subtype)
            else:
                m_type = m.entity_type

            #em = EntityMention(m.id, IntPair(m.start_char, m.end_char+1), m.text, m_type)
            em = EntityMention(m.id, IntPair(m.syn_node.start_token.start_edt, m.syn_node.end_token.end_edt + 1), m.text, m_type)

            head = m.head
            for t in s.tokens:
                #if t.start_char_offset() == head.start_char and t.end_char_offset() == (head.end_char+1):
                if t.start_char_offset() == head.start_token.start_edt and t.end_char_offset() == (head.end_token.end_edt + 1):
                    em.head_token = t
                    break

            doc.add_entity_mention(em)


def add_value_mentions(st, s, doc):
    """
    :type st: serifxml.SentenceTheory
    :type s: nlplingo.text.text_span.Sentence
    :type doc: nlpling.text.text_theory.Document
    """

    if st.value_mention_set is not None:
        for m in st.value_mention_set:
            em = EntityMention(m.id, IntPair(m.start_edt, m.end_edt+1), m.text, m.value_type)
            doc.add_entity_mention(em)


def add_event_anchors_as_entity_mention(serif_sent, doc):
    for event_mention in serif_sent.event_mention_set:
        anchor = event_mention.anchor_node
        start = anchor.start_token.start_edt
        end = anchor.end_token.end_edt + 1
        em = EntityMention(anchor.id, IntPair(start, end), anchor.text, 'ANCHOR')
        doc.add_entity_mention(em)


def add_event_mentions(serif_sent, doc, allow_anchor_as_event_argument=False):
    """
    :type serif_sent: serifxml.Sentence
    :type doc: nlpling.text.text_theory.Document
    """

    if allow_anchor_as_event_argument:
        add_event_anchors_as_entity_mention(serif_sent, doc)

    for event_mention in serif_sent.event_mention_set:
        mention_id = event_mention.id

        # TODO: going forward, we probably want to look at the <EventType> tag for the event type.
        # This begs the question of: what if there are multiple <EventType> tags
        event = Event(mention_id, event_mention.event_type)

        # If the input EventMention has the (semantic_phrase_start, semantic_phrase_end) defined,
        # then we will use those to get the anchor text, instead of using the event_mention.anchor_node.
        # Why? Because (semantic_phrase_start, semantic_phrase_end) allows you to define an arbitrary token span
        # which is not doable with using anchor_node (which is a SynNode).
        # So we assume the user has defined the (semantic_phrase_start, semantic_phrase_end) to override the anchor_node.
        if event_mention.semantic_phrase_start is not None and event_mention.semantic_phrase_end is not None:
            start_index = int(event_mention.semantic_phrase_start)
            end_index = int(event_mention.semantic_phrase_end)
            start = serif_sent.token_sequence[start_index:end_index+1][0].start_edt
            end = serif_sent.token_sequence[start_index:end_index+1][-1].end_edt
            text = serif_sent.get_original_text_substring(start, end)
            event.add_anchor(Anchor(mention_id + '-trigger', IntPair(start, end+1), text, event_mention.event_type))
        else:
            anchor = event_mention.anchor_node
            start = anchor.start_token.start_edt
            end = anchor.end_token.end_edt+1
            text = anchor.text
            event.add_anchor(Anchor(mention_id + '-trigger', IntPair(start, end), text, event_mention.event_type))

        for argument in event_mention.arguments:
            arg_id = argument.value.id      # In Serif land, this could be mention_id, value_mention_id, anchor_node_id
            arg_role = argument.role
            arg_em = doc.get_entity_mention_with_id(arg_id)

            if arg_em is None:
                print('WARNING: arg_em is None for doc={} event_mention.id={} arg_id={}. '
                    'Probably a ValueMention that is only found at the Document-level in SerifXML.'.format(doc.docid, event_mention.id, arg_id))
                continue

            if arg_role.startswith('Time-'):
                arg_role = 'Time'

            event_arg = EventArgument('{}-a{}'.format(mention_id, event.number_of_arguments()), arg_em, arg_role)
            event.add_argument(event_arg)
        doc.add_event(event)

def add_serif_prop_adj_matrices(serif_sent, doc):
    """
    :type serif_sent: serifxml.Sentence
    :type doc: nlpling.text.text_theory.Document
    """
    # print('Sentence Idx', sent_idx)
    assert isinstance(serif_sent, serifxml.Sentence)
    adj_mat = {}
    for sentence_theory in serif_sent.sentence_theories:
        assert isinstance(sentence_theory, serifxml.SentenceTheory)
        if len(sentence_theory.token_sequence) < 1:
            continue
        token_idx_to_token = dict()
        for token_idx, token in enumerate(sentence_theory.token_sequence):
            # @hqiu We need this because original_token_index is not bind correctly!
            assert isinstance(token, serifxml.Token)
            token_idx_to_token[token_idx] = token
        token_to_token_idx = {v: k for k, v in token_idx_to_token.items()}
        for prop_idx, prop in enumerate(sentence_theory.proposition_set):
            # print('Prop Idx', prop_idx)
            assert isinstance(prop, serifxml.Proposition)
            if prop.head is None:
                continue

            pred_head = prop.head
            assert isinstance(pred_head, serifxml.SynNode)
            pred_type = prop.pred_type
            # print(token_to_token_idx[pred_head.end_token])
            if pred_head.text not in adj_mat:
                adj_mat.update({token_to_token_idx[pred_head.end_token]: set()})
            # if prop.head is not None:
            # print("pred_head: {}".format(pred_head.text))
            # print("pred_head_token_idx_set: {}".format({token_to_token_idx[i] for i in pred_head.tokens}))
            # print("pred_type: {}".format(pred_type))

            for prop_arg_idx, prop_arg in enumerate(prop.arguments):
                assert isinstance(prop_arg, serifxml.Argument)
                arg_role = prop_arg.role
                if arg_role == "<ref>":
                    continue
                # print("Arg role: {}".format(arg_role))

                if prop_arg.mention is not None:
                    # is a mention argument
                    mention = prop_arg.mention
                    assert isinstance(mention, serifxml.Mention)
                    # atomic head is also a SynNode
                    # print("mention argument atomic head: {}".format(mention.atomic_head.text))
                    if mention.atomic_head.text not in pred_head.text:
                        # print(mention.atomic_head.end_token)
                        adj_mat[token_to_token_idx[pred_head.end_token]].add(
                            token_to_token_idx[mention.atomic_head.end_token])
                        # adj_mat[token_to_token_idx[pred_head.end_token]].add((mention.atomic_head.text, arg_role));
                if prop_arg.proposition is not None:
                    # is a proposition argument
                    prop_as_prop_arg = prop_arg.proposition
                    assert isinstance(prop_as_prop_arg, serifxml.Proposition)
                    # print("Here you got a child proposition: {}".format(prop_as_prop_arg))
                    if (prop_as_prop_arg.head is not None):
                        if prop_as_prop_arg.head.text not in pred_head.text:
                            # print(prop_as_prop_arg.head.end_token)
                            adj_mat[token_to_token_idx[pred_head.end_token]].add(
                                token_to_token_idx[prop_as_prop_arg.head.end_token])
                if prop_arg.syn_node is not None:
                    prop_as_prop_arg = prop_arg.syn_node
                    assert isinstance(prop_as_prop_arg, serifxml.SynNode)

                else:
                    dir(prop_arg)

    edge_list = []
    for i in adj_mat:
        if adj_mat[i]:
            for j in adj_mat[i]:
                edge_list.append([i, j])
    doc.add_adj_mat(edge_list)

def add_serif_dependency_matrices(serif_sent, doc):
    """
    :type serif_sent: serifxml.Sentence
    :type doc: nlpling.text.text_theory.Document
    """
    # print('Sentence Idx', sent_idx)
    assert isinstance(serif_sent, serifxml.Sentence)
    adj_mat = {}
    dep_rels = {}
    head_array = None
    dep_rels_final = None
    assert(len(serif_sent.sentence_theories) == 1)
    for sentence_theory in serif_sent.sentence_theories:
        assert isinstance(sentence_theory, serifxml.SentenceTheory)
        head_array = [0] * len(sentence_theory.token_sequence)
        dep_rels_final = [DEPREL_TO_ID['no_label']] * len(sentence_theory.token_sequence)
        token_idx_to_token = dict()
        for token_idx, token in enumerate(sentence_theory.token_sequence):
            # @hqiu We need this because original_token_index is not bind correctly!
            assert isinstance(token, serifxml.Token)
            token_idx_to_token[token_idx] = token

        token_to_token_idx = {v: k for k, v in token_idx_to_token.items()}
        # print(token_to_token_idx)
        for prop_idx, prop in enumerate(sentence_theory.dependency_set):
            # print('Prop Idx', prop_idx)
            assert isinstance(prop, serifxml.Proposition)
            # print(prop.head)
            assert(prop.head is not None)
            pred_head = prop.head
            assert isinstance(pred_head, serifxml.SynNode)
            pred_type = prop.pred_type
            # print(token_to_token_idx[pred_head.end_token])
            head_idx = token_to_token_idx[pred_head.end_token]
            if head_idx not in adj_mat:
                adj_mat.update({head_idx : set()})
            # if prop.head is not None:
            # print("pred_head: {}".format(pred_head.text))
            # print("pred_head_token_idx_set: {}".format({token_to_token_idx[i] for i in pred_head.tokens}))
            # print("pred_type: {}".format(pred_type))

            for prop_arg_idx, prop_arg in enumerate(prop.arguments):
                assert isinstance(prop_arg, serifxml.Argument)
                arg_role = prop_arg.role
                # print(arg_role)
                # if arg_role == "<ref>":
                #    continue
                # print("Arg role: {}".format(arg_role))

                if prop_arg.mention is not None:
                    # is a mention argument
                    mention = prop_arg.mention
                    assert isinstance(mention, serifxml.Mention)
                    # atomic head is also a SynNode
                    # print("mention argument atomic head: {}".format(mention.atomic_head.text))
                    if mention.atomic_head.text not in pred_head.text:
                        # print(mention.atomic_head.end_token)
                        arg_idx = token_to_token_idx[mention.atomic_head.end_token]
                        adj_mat[head_idx].add(arg_idx)
                        head_array[arg_idx] = head_idx + 1
                        if arg_idx not in dep_rels:
                            dep_rels[arg_idx] = []
                        dep_rels[arg_idx].append(arg_role)

                        # adj_mat[token_to_token_idx[pred_head.end_token]].add((mention.atomic_head.text, arg_role));
                if prop_arg.proposition is not None:
                    # is a proposition argument
                    prop_as_prop_arg = prop_arg.proposition
                    assert isinstance(prop_as_prop_arg, serifxml.Proposition)
                    # print("Here you got a child proposition: {}".format(prop_as_prop_arg))
                    if (prop_as_prop_arg.head is not None):
                        # print('prop head end token', prop_as_prop_arg.head.end_token)
                        arg_idx = token_to_token_idx[prop_as_prop_arg.head.end_token]
                        head_array[arg_idx] = head_idx + 1
                        adj_mat[head_idx].add(arg_idx)
                        if arg_idx not in dep_rels:
                            dep_rels[arg_idx] = []
                        dep_rels[arg_idx].append(arg_role)
                if prop_arg.syn_node is not None:
                    prop_as_prop_arg = prop_arg.syn_node
                    assert isinstance(prop_as_prop_arg, serifxml.SynNode)
                    syn_head = prop_as_prop_arg.head
                    # print(syn_head.end_token)
                    # print(token_to_token_idx[syn_head.end_token])
                    arg_idx = token_to_token_idx[syn_head.end_token]
                    adj_mat[head_idx].add(arg_idx)
                    head_array[arg_idx] = head_idx + 1
                    # adj_mat[token_to_token_idx[pred_head.end_token]].add((mention.atomic_head.text, arg_role));
                    if arg_idx not in dep_rels:
                        dep_rels[arg_idx] = []
                    dep_rels[arg_idx].append(arg_role)
                    # atomic head is also a SynNode
                    #print("mention argument atomic head: {}".format(mention.atomic_head.text))
                    #if mention.atomic_head.text not in pred_head.text:
                    #    # print(mention.atomic_head.end_token)
                    #    adj_mat[token_to_token_idx[pred_head.end_token]].add(
                    #        token_to_token_idx[mention.atomic_head.end_token])
                        # adj_mat[token_to_token_idx[pred_head.end_token]].add((mention.atomic_head.text, arg_role));
                else:
                    dir(prop_arg)

    truth_checks_head_array = [head == 0 for head in head_array]
    assert(only1(truth_checks_head_array))
    # tree = head_to_tree(head_array, None, 1)

    # adj = tree_to_adj(len(head_array), tree, directed=False, self_loop=False)
    # pprint_tree(tree)
    #edge_list = []
    #for i in adj_mat:
    #    if adj_mat[i]:
    #        for j in adj_mat[i]:
    #            edge_list.append([i, j])

    # final_dep_rels = {}
    for i in dep_rels:
        assert(len(dep_rels[i]) == 1)
        destroy_subtype = dep_rels[i][0].split(':')[0]
        # print(dep_rels[i][0].split(':')[0])
        assert(destroy_subtype in DEPREL_TO_ID)
        dep_rels_final[i] = DEPREL_TO_ID[destroy_subtype]

    truth_checks_dep_rels = [dep_rel == DEPREL_TO_ID['no_label'] for dep_rel in dep_rels_final]
    assert(only1(truth_checks_dep_rels))
    doc.add_adj_mat((head_array, dep_rels_final))


def to_lingo_sentence(serif_doc, st_index, sentence, lingo_doc, add_serif_entity_mentions=True, add_serif_event_mentions=False, add_serif_entity_relation_mentions=False,
                      add_serif_prop_adj=False, add_serif_dep_graph=False, allow_anchor_as_event_argument=False):
    """
    :param serif_doc: serifxml.Document
    :param st_index: int
    :param sentence: serifxml.Sentence
    :param lingo_doc: nlplingo.text.text_theory.Document
    :param add_serif_entity_mentions:
    :param add_serif_event_mentions:
    :param add_serif_entity_relation_mentions:
    :return:nlplingo.text.text_theory.Document
    """
    docid = lingo_doc.docid
    st = sentence.sentence_theories[0]
    """:type: serifxml.SentenceTheory"""
    if len(st.token_sequence) == 0:
        s = Sentence(docid, IntPair(sentence.start_edt, sentence.end_edt + 1), '', [], st_index)
        lingo_doc.add_sentence(s)
        return

    st_text, st_start, st_end = get_snippet(serif_doc,st)

    tokens = to_tokens(st)
    assert st_start == tokens[0].start_char_offset()
    assert (st_end + 1) == tokens[-1].end_char_offset()

    s = Sentence(docid, IntPair(st_start, st_end + 1), st_text, tokens, st_index)
    if add_serif_entity_mentions:
        add_entity_mentions(st, s, lingo_doc)
        add_value_mentions(st, s, lingo_doc)
        add_names(st, lingo_doc)
    if add_serif_event_mentions and sentence.event_mention_set:
        add_event_mentions(sentence, lingo_doc, allow_anchor_as_event_argument)

    if add_serif_entity_relation_mentions:
        for serif_relation_mention in sentence.rel_mention_set:
            arg1_serif = serif_relation_mention.left_mention
            arg2_serif = serif_relation_mention.right_mention
            arg1_lingo = lingo_doc.get_entity_mention_with_id(arg1_serif.id)
            arg2_lingo = lingo_doc.get_entity_mention_with_id(arg2_serif.id)
            relation_type = serif_relation_mention.type
            entity_relation = EntityRelation(relation_type, arg1_lingo, arg2_lingo)
            lingo_doc.add_entity_entity_relation(entity_relation)

    if add_serif_prop_adj:
        add_serif_prop_adj_matrices(sentence, lingo_doc)

    if add_serif_dep_graph:
        add_serif_dependency_matrices(sentence, lingo_doc)

    lingo_doc.add_sentence(s)
    return

def to_lingo_doc(filepath, add_serif_entity_mentions=True, add_serif_event_mentions=False, add_serif_eer=False, add_serif_entity_relation_mentions=False,
                 add_serif_prop_adj=False, add_serif_dep_graph=False, allow_anchor_as_event_argument=False):
    """Takes in a filepath to a SerifXML, and use its sentences, tokens, entity-mentions, value-mentions
    to construct a nlplingo.text.text_theory.Document
    Returns: nlplingo.text.text_theory.Document
    """
    serif_doc = serifxml.Document(filepath)
    """:type: serifxml.Document"""

    docid = serif_doc.docid
    lingo_doc = lingoDoc(docid)
    for st_index, sentence in enumerate(serif_doc.sentences):
        to_lingo_sentence(serif_doc,st_index, sentence, lingo_doc = lingo_doc, add_serif_entity_mentions = add_serif_entity_mentions, add_serif_event_mentions = add_serif_event_mentions,\
                          add_serif_entity_relation_mentions = add_serif_entity_relation_mentions, add_serif_prop_adj=add_serif_prop_adj, add_serif_dep_graph=add_serif_dep_graph, allow_anchor_as_event_argument=allow_anchor_as_event_argument)

    # read event-event relation mentions
    for serif_eerm in serif_doc.event_event_relation_mention_set or []:
        serif_em_arg1 = None
        serif_em_arg2 = None
        for arg in serif_eerm.event_mention_relation_arguments:
            if arg.role == "arg1":
                serif_em_arg1 = arg.event_mention
            if arg.role == "arg2":
                serif_em_arg2 = arg.event_mention
        if not serif_em_arg1 or not serif_em_arg2:
            print("EER: could not find two arguments for causal relation!")
            continue

        lingo_em_arg1 = lingo_doc.get_event_with_id(serif_em_arg1.id)
        lingo_em_arg2 = lingo_doc.get_event_with_id(serif_em_arg2.id)
        relation_type = serif_eerm.relation_type
        eer = EventEventRelation(relation_type, lingo_em_arg1, lingo_em_arg2)
        lingo_doc.add_event_event_relation(eer)

    return lingo_doc


# TODO: add lemma dict, train_test.read_doc_annotation() method
if __name__ == "__main__":

    serifxml_filepath = sys.argv[1]
    lingo_doc = to_lingo_doc(serifxml_filepath)


    # for s in lingo_doc.sentences:
    #     token_strings = []
    #     for token in s.tokens:
    #         token_strings.append('{}-{}:{}/{}'.format(token.start_char_offset(), token.end_char_offset(), token.text, token.pos_tag))
    #     print(' '.join(token_strings))

