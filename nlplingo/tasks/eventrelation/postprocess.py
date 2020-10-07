from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging

from serifxml3 import construct_joint_prop_between_events, prune_bad_patterns_from_joint_prop_tree
import serifxml3

logger = logging.getLogger(__name__)

bad_set = {'gives', 'give', 'gave', 'given', 'giving', 'take', 'took', 'taken', 'takes', 'taking', 'make', 'made', 'makes',
        'making', 'seem', 'seems', 'seeming', 'seemed', 'put', 'indicates', 'indicated', 'indicating', 'continues', 'facing',
        'continued', 'continuing', 'continue', 'said', 'contributed', 'contributes', 'contributing', 'contribute', 'caused', 'causes', 'cause', 'causing', 'resulted', 'result', 'results', 'resulting', 'affect',  'affected', 'affects', 'affecting', 'promoting', 'promote', 'promotes', 'promoted', 'rely', 'added', 'focus', 'expected', 'expecting', 'led', 'lead', 'leads', 'leading', 'provided', 'provide', 'provides', 'providing', 'becoming', 'becomes', 'become', 'became', 'address', 'involved', 'needed', 'needs', 'needing', 'need', 'help', 'helps', 'helping', 'helped', 'appear', 'created', 'prevents', 'prevent', 'prevented', 'attributed', 'proved', 'remain', 'remained', 'remains', 'remaining', 'tended', 'tend', 'tends', 'tending', 'play', 'plays', 'played', 'playing', 'follows', 'raised', 'relate', 'relates', 'related', 'relating', 'expressed', 'assumed', 'argues', 'aims', 'includes', 'include', 'including', 'included', 'noted', 'regarding', 'occurred', 'described', 'describes', 'very', 'merely', 'according', 'According', 'create', 'created', 'creates', 'addition', 'culminate', 'referred', 'poured', 'support', 'meant', 'believe', 'think', 'says', 'involve', 'places', 'show', 'expected', 'barred', 'overstepped', 'say', 'restrictions', 'mean', 'saying', 'refers', 'forecast', 'live', 'severe', 'draws', 'serious', 'running', 'sought', 'estimate', 'concerning', 'lasted', 'voiced', 'estimated', 'enhance', 'broke', 'exacting', 'exaggerated', 'seen', 'going', 'caught', 'go', 'admit', 'opposed', 'stated', 'set', 'handed', 'appears', 'appeared', 'boasted', 'know', 'pose', 'facilitate', 'stop', 'mitigate', 'believes', 'rejected', 'looking', 'encourage', 'starts', 'indicate', 'adds', 'accelerate', 'sound', 'thought', 'predict', 'arguing', 'turn', 'stimulate', 'fix', 'proven', 'puts', 'projects', 'thinking', 'share', 'placed', 'look', 'state', 'asked', 'asks', 'ask'}


bad_set_ldc = {'established', 'substituted', 'narrowed', 'translates', 'culminated', 'links', 'limits', 'deter', 'deters', 'escalating', 'impacting', 'compelled', 'gives', 'turning', 'fuelling', 'assisting', 'addressing', 'waiting', 'reflected', 'based', 'rising', 'produced', 'leaves', 'get', 'forced', 'initiated', 'perpetuated', 'gained', 'threatened', 'called', 'winning', 'downsized', 'sharpening', 'deterred', 'turned', 'add', 'stemming', 'poses', 'depending', 'enhanced', 'stepping', 'complicated', 'eroded', 'curtailed', 'foundered', 'this', 'worsened', 'building', 'raising', 'emerged', 'resulting', 'reasons', 'easing', 'cuts', 'illustrated', 'acknowledges', 'deteriorated', 'follow', 'determined', 'portioned', 'complementing', 'achieved', 'strengthened', 'plays', 'pushes', 'attributes', 'preferred', 'permits', 'accompanied', 'raises', 'presented', 'advances', 'stemmed', 'obtained', 'required', 'bringing', 'increasing', 'involving', 'compounded', 'results', 'increases', 'caused', 'creating', 'slowing', 'translated', 'induced', 'contradicts', 'accounted', 'using', 'restricted', 'maintains', 'degenerates', 'furthered', 'inspired', 'contained', 'retaliated', 'controlled', 'aided', 'triggered', 'promotes', 'enhancing', 'restored', 'blocked', 'aggravated', 'instigated', 'denies', 'as', 'encouraging', 'carried', 'prevented', 'undermining', "don't", 'preventing', 'affects', 'address', 'advancing', 'manifested', 'delayed', 'aimed', 'goaded', 'have', 'stoked', 'sparked', 'defused', 'underscored', 'underpinned', 'reaffirmed', 'shaped', 'related', 'exerted', 'creates', 'furthering', 'permitting', 'causes', 'overcoming', 'fueling', 'ongoing', 'facilitated', 'sparking', 'worsening', 'depends', 'constraining', 'ensuring', 'managed', 'evolved', 'risked', 'jumped', 'linking', 'extinguished', 'supplanted', 'enabled', 'reaching', 'having', 'averted', 'charging', 'enjoys', 'accumulated', 'grew', 'determines', 'remobilised', 'replace', 'attributing', 'strengthening', 'secured', 'garnered', 'drives', 'gave', 'did', 'less', 'swelled', 'stipulated', 'enables', 'agreed', 'impeded', 'forfeiting', 'disrupting', 'jeopardizing', 'multiplied', 'responding', 'affected', 'discouraging', 'causing', 'arising', 'ending', 'accusing', 'weakening', 'accommodates', 'challenged', 'followed', 'leads', 'prolonging', 'added', 'diminished', 'stymied', 'roots', 'created', 'costs', 'cited', 'subsided', 'doubled', 'stopped', 'compounding', 'promoting', 'requires', 'implies', 'reflects', 'challenges', 'generating', 'linked', 'sharpened', 'delivered', 'outpacing', 'engendered', 'reduce', 'deteriorates', 'instilling', 'ushered', 'seeks', 'obliged', 'dictated', 'jeopardising', 'posed', 'utilizing', 'rendering', 'blocking', 'resisted', 'collapsed', 'stems', 'expressed', 'assuring', 'deepen', 'managing', 'became', 'presupposes', 'nurtured', 'enabling', 'scuttling', 'widened', 'allow', 'during', 'embarking', 'mitigating', 'improved', 'intensified', 'reinforcing', 'learned', 'leave', 'establish', 'constrained', 'calls', 'fosters', 'undermined', 'blamed', 'offered', 'were', 'hamstringing', 'mitigates', 'conditioned', 'exposes', 'characterized', 'erodes', 'originates', 'aggravating', 'kicks', 'exacerbates', 'ascribed', 'adding', 'prompted', 'needs', 'hurdles', 'capitalizes', 'leading', 'follows', 'equipped', 'emboldened', 'directed', 'improving', 'relies', 'resolved', 'mounting', 'supported', 'inherited', 'contributing', 'deepened', 'involves', 'remains', 'given', 'ensure', 'aroused', 'following', 'influenced', 'impeding', 'demonstrated', 'combined', 'relied', 'justified', 'plugging', 'reflecting', 'promoted', 'give', 'relegating', 'generated', 'improves', 'increased', 'accounting', 'lost', 'reduces', 'problems', 'drains', 'destroying', 'facilitating', 'made', 'protecting', 'including', 'started', 'capitalizing', 'leverages', 'protected', 'imposed', 'triggering', 'waned', 'benefited', 'shutting', 'escalated', 'melting', 'heightened', 'providing', 'making', 'threatens', 'improve', 'lowering', 'towards', 'alleviating', 'included', 'predicated', 'hindered', 'tripled', 'giving', 'provides', 'contributes', 'obstructed', 'derives', 'fuelled', 'anchored', 'crippling', 'reinforced', 'scrambled', 'has', 'manifesting', 'served', 'complements', 'devastating', 'relieving', 'sustained', 'stepped', 'accused', 'builds', 'see', 'bolstering', 'accompanying', 'driving', 'accords', 'prioritizes', 'influences', 'deepens', 'descended', 'faces', 'revolving', 'reasserting', 'escalates', 'hampered', 'reinforces', 'motivating', 'forestalling', 'licting', 'attributed', 'dimmed', 'conducing', 'exaserbated', 'explained', 'calling', 'opening', 'fuels', 'makes', 'drawing', 'boosted', 'spurred', 'slashed', 'requirements', 'culminating', 'rallying', 'exacerbated', 'hampers', 'helped', 'scaled', 'allows', 'reached', 'precipitated', 'accelerated', 'motives', 'keeps', 'exacerbating', 'develop', 'resulted', 'spawned', 'impacted', 'intensifying', 'assists', 'changed', 'necessitates', 'quelled', 'rises', 'prevents', 'complicates', 'needed', 'marred', 'capitalized', 'fueled', 'prohibited', 'dwindled', 'led', 'used', 'growing', 'supports', 'geared', 'allowed', 'augmenting', 'ended', 'fostering', 'indicating', 'subjected', 'kicked', 'supplements', 'eroding', 'hopes', 'disrupted', 'suspended', 'limited', 'initiating', 'eschewing', 'thus', 'unleashed', 'inflamed', 'provided', 'paving', 'challenging', 'associated', 'accrued', 'leaving', 'eliminating', 'engenders', 'fanned', 'faded', 'represents', 'slow', 'helps', 'pushed', 'alleviated', 'skyrocketed', 'raised', 'betrayed', 'ensures', 'citing', 'degrading', 'denied', 'declining', 'responded', 'unless', 'halted', 'empowers', 'representing', 'deteriorating', 'make', 'owing', 'tied', 'dealing', 'informed', 'eased', 'triggers', 'grappled', 'provoked', 'enhances', 'opened', 'receded', 'generates', 'countering', 'encouraged', 'presents', 'attracting', 'overriding', 'reducing', 'keep', 'awaits', 'conducted', 'surrounding', 'forcing', 'declined', 'accounts', 'allowing', 'rooted', 'originated', 'forces', 'stressing', 'continued', 'helping', 'requiring', 'restricting', 'illustrates', 'undermines', 'derived', 'clouded', 'characterised', 'involved', 'limiting', 'deepening', 'contributed', 'owed', 'reduced', 'weakened', 'bring', 'brought', 'depend', 'believed'}

word_pairs = {'said#a statement', 'said#statement', 'life#sentenced', 'head#boost', 'growth#decelerate',
              'price#decelerate', 'investment#total', 'estimated#cost', 'open#trial', 'flights#open',
              'flight#open', 'death#saved', 'respect#danger', 'birr#spent', 'start#trial', 'begin#trial',
              'famine#saved', 'violence#spared', 'send#fled', 'hold#conference', 'confidence#arrive',
              'start#rebelled', 'scatter#fled', 'alleged#died', 'staying#fled'}

betwixt_phrase = {'will', 'has', 'have', 'should', 'would', 'could', 'had', 'to have', 'would have', 'that',
                  'that have', 'it would', 'gave', 'continues to', 'is expected to', 'that will', 'that would',
                  'was', 'that put', 'that could', 'will not', 'that it will', 'growth will',
                  'will most likely', 'will considerably', 'would also', 'would not have', 'would be',
                  'to prevent and', 'to prevent or', 'will not affect the', 'came to', 'has not',
                  'were abruptly', 'that this was going to turn into', 'expressed', 'by boosting'}

bad_patterns = [('tail-head', 'spur', 'Cause-Effect', 'spurred by'),
                ('head-tail', 'spurred by', 'Catalyst-Effect'),
                ('head-tail', 'spurred by', 'Cause-Effect'),
                ('head-tail', 'is attributed to', 'Cause-Effect'),
                ('head-tail', 'is attributed to', 'Catalyst-Effect'),
                ('tail-head', 'increased', 'Catalyst-Effect', 'increased by'),
                ('tail-head', 'leads to', 'Before-After'),
                ('tail-head', 'followed by', 'Before-After'),
                ('tail-head', 'caused by', 'Cause-Effect'),
                ('head-tail', "would n't", 'Catalyst-Effect'),
                ('head-tail', "failed", 'Cause-Effect'),
                ('head-tail', "failed", 'Preventative-Effect'),
                ('head-tail', "not resulted", 'Cause-Effect'),
                ('head-tail', "was sparked by", 'Cause-Effect'),
                ('tail-head', "spread", 'Cause-Effect', 'spread by'),
                ('tail-head', "improves", 'Catalyst-Effect'),
                ('head-tail', "should not", 'Cause-Effect'),
                ('head-tail', "caused no", 'Cause-Effect'),
                ('head-tail', "did n't cause", 'Cause-Effect'),
                ('tail-head', "pushed", 'Cause-Effect', 'pushed by'),
                ('tail-head', "has been reduced to", 'MitigatingFactor-Effect'),
                ('head-tail', "how to prevent", 'Preventative-Effect'),
                ('tail-head', "sparked", 'Cause-Effect', 'sparked by'),
                ('tail-head', "could provide momentum", 'Cause-Effect'),
                ('head-tail', "neither increases", 'Catalyst-Effect'),
                ('head-tail', "stopped after", 'Preventative-Effect')
                ]

def prefilter(indices, examples):
    for i, example in enumerate(examples):
        arg1_text = example.anchor1.text
        arg2_text = example.anchor2.text
        sentence_text = example.sentence.text

        # print('arg1_text', arg1_text)
        # print('arg2_text', arg2_text)
        # print('sentence', sentence_text)

        # Label overlaps as NA
        if arg1_text in arg2_text or arg2_text in arg1_text:
            indices.remove(i)
            continue

        lower_arg1_text = arg1_text.lower()
        lower_arg2_text = arg2_text.lower()
        # Skip 'bad_set'
        if (lower_arg1_text in bad_set) or (lower_arg2_text in bad_set) or (lower_arg1_text in bad_set_ldc) or (
                lower_arg2_text in bad_set_ldc):
            indices.remove(i)
            continue

        # Block certain event pairs
        word_pair = arg1_text + '#' + arg2_text

        word_pair_found = False
        for pair in word_pairs:
            if word_pair == pair:
                word_pair_found = True
                break

        if word_pair_found:
            indices.remove(i)
            continue

        context_pair_set = set()

        for phrase in betwixt_phrase:
            context_pair_set.add(arg1_text + ' ' + phrase + ' ' + arg2_text)
            context_pair_set.add(arg2_text + ' ' + phrase + ' ' + arg1_text)

        context_pair_found = False
        for context_pair in context_pair_set:
            if context_pair in sentence_text:
                context_pair_found = True
                break

        if context_pair_found:
            indices.remove(i)
            continue

        # Serif removal operations
        # print('joint_serif_prop_tree')
        joint_prop_tree, e_to_o_map = construct_joint_prop_between_events(example.serif_sentence, example.serif_event_0, example.serif_event_1)
        example.joint_serif_prop_tree = joint_prop_tree
        # is_bad = False
        is_bad = prune_bad_patterns_from_joint_prop_tree(example.joint_serif_prop_tree, e_to_o_map)
        # print('bad_pattern', is_bad)

        if is_bad:
            indices.remove(i)
            continue


def check_bad_pattern(bad_patterns, relation_type, slot1end, slot0start, sentence_text):
    tail_head = slot1end < slot0start
    for pattern in bad_patterns:
        if (pattern[0] == 'tail-head' and tail_head) \
                or (pattern[0] == 'head-tail' and not tail_head):
            smaller = min(slot0start, slot1end)
            larger = max(slot0start, slot1end)
            tokenized_arr = sentence_text
            in_between = ' '.join(tokenized_arr[smaller:larger + 1])
            if pattern[1] in in_between:
                if len(pattern) >= 3:
                    if relation_type == pattern[2]:
                        if len(pattern) >= 4:
                            if pattern[3] not in in_between:
                                return True
                            else:
                                return False
                        else:
                            return True
                else:
                    return True
    return False

def compute_distance(sentence_text, slot0start, slot0end, slot1start, slot1end):
    smaller = min(slot0start, slot1start)
    larger = max(slot0end, slot1end)
    span_str = sentence_text[smaller:larger+1]
    num_in_between = len(span_str.split()) - 2 # count number of tokens in between the head/tail words
    if num_in_between <= 0:
        return 0
    else:
        return num_in_between

def postfilter(indices, examples, predictions, event_domain, none_class_index, threshold, all_eer_predictions, model_name):
    if model_name == 'cnn' or model_name == 'cnn_opt':
        model_name = 'nn_model2'
    elif model_name == 'bert_mention':
        model_name = 'nn_model1'
    else:
        raise Exception('model_name not supported'
                        )
    for i, prediction in enumerate(predictions):
        relation_type = prediction[0]
        # remove NA predictions
        if none_class_index == event_domain.get_eer_type_index(relation_type):
            indices.remove(i)
            continue

        # Skip a low-precision type
        if relation_type == 'Precondition-Effect' or relation_type == 'Before-After':
            indices.remove(i)
            continue

        confidence = float(prediction[1])
        if confidence <= threshold:
            indices.remove(i)
            continue

        arg1_text = examples[i].anchor1.text
        arg2_text = examples[i].anchor2.text
        sentence_text = examples[i].sentence.text
        
        sentence_start = examples[i].sentence.start_char_offset()
        slot0start =  examples[i].anchor1.start_char_offset() - sentence_start
        slot1end = examples[i].anchor2.end_char_offset() - sentence_start

        # Block certain event-event-relation triplets
        if (arg1_text == 'jail' and arg2_text == 'sentenced' and relation_type == 'Cause-Effect') or \
                (arg1_text == 'prison' and arg2_text == 'sentenced' and relation_type == 'Cause-Effect') or \
                (arg1_text == 'military' and arg2_text == 'operations' and relation_type == 'MitigatingFactor-Effect') or \
                (arg1_text == '50 counts' and arg2_text == 'charged' and relation_type == 'Cause-Effect') or \
                (arg1_text == 'killed' and arg2_text == 'attack' and relation_type == 'Cause-Effect') or \
                (arg1_text == 'imprisonment' and arg2_text == 'sentenced' and relation_type == 'Cause-Effect') or \
                (arg1_text == 'efforts' and arg2_text == 'witnesses' and relation_type == 'Catalyst-Effect'):
            indices.remove(i)
            continue

        if check_bad_pattern(bad_patterns, relation_type, slot1end, slot0start, sentence_text):
            indices.remove(i)
            continue

        # add prediction information to all_eer_predictions
        eer_key = construct_eer_key(examples[i])
        if eer_key not in all_eer_predictions:
            all_eer_predictions[eer_key] = {'LearnIt' : list(), 'nn_model1' : list(), 'nn_model2' : list()}

        # adjust confidence based on token distance between head and tail
        # decrease probability if head / tail are far away
        slot0end = examples[i].anchor1.end_char_offset() - sentence_start
        slot1start =  examples[i].anchor2.start_char_offset() - sentence_start
        confidence = confidence * (pow(0.994, compute_distance(sentence_text, slot0start, slot0end, slot1start, slot1end)))

        # adjust confidence according to Serif tree distance
        if examples[i].joint_serif_prop_tree is None:
            confidence = confidence * 0.95
        # examples[i].joint_serif_prop_tree

        all_eer_predictions[eer_key][model_name].append((confidence, relation_type, examples[i]))


def postfilter_general(indices, examples, predictions, event_domain, none_class_index, threshold, all_eer_predictions,
               model_name):
    model_name = 'nn_model1' # TODO: make this cleaner; this is a bit of a hack to match with consolidation logic...
    for i, prediction in enumerate(predictions):
        relation_type = prediction[0]
        # remove NA predictions
        if none_class_index == event_domain.get_eer_type_index(relation_type):
            indices.remove(i)
            continue

        confidence = float(prediction[1])
        if confidence <= threshold:
            indices.remove(i)
            continue

        sentence_text = examples[i].sentence.text

        sentence_start = examples[i].sentence.start_char_offset()
        slot0start = examples[i].anchor1.start_char_offset() - sentence_start
        slot1end = examples[i].anchor2.end_char_offset() - sentence_start

        # add prediction information to all_eer_predictions
        eer_key = construct_eer_key(examples[i])
        if eer_key not in all_eer_predictions:
            all_eer_predictions[eer_key] = {'LearnIt': list(), 'nn_model1': list(), 'nn_model2': list()}

        # adjust confidence based on token distance between head and tail
        # decrease probability if head / tail are far away
        slot0end = examples[i].anchor1.end_char_offset() - sentence_start
        slot1start = examples[i].anchor2.start_char_offset() - sentence_start
        confidence = confidence * (
            pow(0.994, compute_distance(sentence_text, slot0start, slot0end, slot1start, slot1end)))

        all_eer_predictions[eer_key][model_name].append((confidence, relation_type, examples[i]))

def construct_eer_key(example):
    # nlplingo example
    eer_key = example.sentence.docid + '#' + example.sentence.int_pair.to_string() + '#' + example.anchor1.int_pair.to_string() + '#' + example.anchor2.int_pair.to_string()
    return eer_key

def construct_rev_key(key):
    key_split = key.split('#')
    reverse_key = '#'.join([key_split[3], key_split[2]])
    reverse_key = '#'.join([key_split[0], key_split[1]]) + '#' + reverse_key
    return reverse_key

def sent_highlighter(sentence, anchor1, anchor2):
    """
    :param sentence: SentencePrediction
    :param anchor1: TriggerPrediction
    :param anchor2: TriggerPrediction
    :return:
    """
    anchor1_index_start = anchor1.start - sentence.start
    anchor1_index_end = anchor1.end - sentence.start + 1
    anchor2_index_start = anchor2.start - sentence.start
    anchor2_index_end = anchor2.end - sentence.start + 1

    if anchor1_index_start < anchor2_index_start:
        min_offset = (anchor1_index_start, anchor1_index_end)
        max_offset = (anchor2_index_start, anchor2_index_end)
        min_key = ' [HEAD] '
        max_key = ' [TAIL] '
    else:
        max_offset = (anchor1_index_start, anchor1_index_end)
        min_offset = (anchor2_index_start, anchor2_index_end)
        max_key = ' [HEAD] '
        min_key = ' [TAIL] '
    new_sentence = ""
    sentence = sentence.text
    new_sentence += sentence[:min_offset[0]] + min_key + sentence[min_offset[0]:min_offset[1]] + min_key + \
                    sentence[min_offset[1]: max_offset[0]] + max_key + sentence[max_offset[0]:max_offset[1]] + max_key + \
                    sentence[max_offset[1]: -1]
    return new_sentence

def print_relation_from_eer_key(eer_key, sent_edt_off_to_sent_dict, is_rev=False):
    """
    :param eer_key: str constructed from NLPLINGO (not Serif!) offsets
    :param sent_edt_off_to_sent_dict: dictionary mapping docids to sentences encoded by sentence offset (Serif format) within the docid
    """
    key_split = eer_key.split('#')
    docid = key_split[0]
    (sentence_start, sentence_end) = eval(key_split[1])
    sentence = sent_edt_off_to_sent_dict[docid][sentence_start, sentence_end - 1].text
    anchor1_indices = eval(key_split[2])
    anchor2_indices = eval(key_split[3])

    anchor1_index_start = anchor1_indices[0] - sentence_start
    anchor1_index_end = anchor1_indices[1] - sentence_start + 1
    anchor2_index_start = anchor2_indices[0] - sentence_start
    anchor2_index_end = anchor2_indices[1] - sentence_start + 1

    if anchor1_index_start < anchor2_index_start:
        min_offset = (anchor1_index_start, anchor1_index_end)
        max_offset = (anchor2_index_start, anchor2_index_end)
        min_key = ' [HEAD] '
        max_key = ' [TAIL] '
    else:
        max_offset = (anchor1_index_start, anchor1_index_end)
        min_offset = (anchor2_index_start, anchor2_index_end)
        max_key = ' [HEAD] '
        min_key = ' [TAIL] '
    new_sentence = ""
    new_sentence += sentence[:min_offset[0]] + min_key + sentence[min_offset[0]:min_offset[1]] + min_key + \
                    sentence[min_offset[1]: max_offset[0]] + max_key + sentence[max_offset[0]:max_offset[1]] + max_key + \
                    sentence[max_offset[1]: -1]

    if is_rev:
        logging.debug('START OF NEW RELATION %s %s', docid, new_sentence)
    else:
        logging.debug('START OF NEW RELATION %s %s', docid, new_sentence)

def add_serif_eerm_to_all_eer_predictions(all_eer_predictions, serif_eerm, lingo_doc):
    model = serif_eerm.model

    if model != 'LearnIt':
        raise Exception('This should be a LearnIt relation.')
    else:
        serif_em_arg1 = None
        serif_em_arg2 = None
        for arg in serif_eerm.event_mention_relation_arguments:
            if arg.role == "arg1":
                serif_em_arg1 = arg.event_mention
            if arg.role == "arg2":
                serif_em_arg2 = arg.event_mention
        if not serif_em_arg1 or not serif_em_arg2:
            logging.info("EER: could not find two arguments for causal relation!")
            exit(1)

        # Serif removal operations
        # print('joint_serif_prop_tree')
        serif_sentence = serif_em_arg1.owner_with_type(serifxml3.Sentence)
        joint_prop_tree, e_to_o_map = construct_joint_prop_between_events(serif_sentence, serif_em_arg1, serif_em_arg2)
        # is_bad = False
        is_bad = prune_bad_patterns_from_joint_prop_tree(joint_prop_tree, e_to_o_map)
        # print('bad_pattern', is_bad)

        if is_bad:
            return

        lingo_em_arg1 = lingo_doc.get_event_with_id(serif_em_arg1.id)
        lingo_em_arg2 = lingo_doc.get_event_with_id(serif_em_arg2.id)
        found_sent = False

        for sentence in lingo_doc.sentences:
            if lingo_em_arg1.overlaps_with_anchor(sentence) and lingo_em_arg2.overlaps_with_anchor(sentence):
                eer_key = sentence.docid + '#' + sentence.int_pair.to_string() + '#' + lingo_em_arg1.anchors[0].int_pair.to_string() + '#' + lingo_em_arg2.anchors[0].int_pair.to_string()
                found_sent = True
                break
        if not found_sent:
            raise Exception('No sentence found for serif_eerm.')
        relation_type = serif_eerm.relation_type
        pattern = serif_eerm.pattern

        # add prediction information to all_eer_predictions
        if eer_key not in all_eer_predictions:
            all_eer_predictions[eer_key] = {'LearnIt' : list(), 'nn_model1' : list(), 'nn_model2' : list()}
        all_eer_predictions[eer_key]['LearnIt'].append((relation_type, serif_eerm, pattern))

def construct_eer_json(serif_eerm, lingo_doc):
    model = serif_eerm.model

    if model != 'LearnIt':
        raise Exception('This should be a LearnIt relation.')
    else:
        serif_em_arg1 = None
        serif_em_arg2 = None
        for arg in serif_eerm.event_mention_relation_arguments:
            if arg.role == "arg1":
                serif_em_arg1 = arg.event_mention
            if arg.role == "arg2":
                serif_em_arg2 = arg.event_mention
        if not serif_em_arg1 or not serif_em_arg2:
            logging.info("EER: could not find two arguments for causal relation!")
            exit(1)
        lingo_em_arg1 = lingo_doc.get_event_with_id(serif_em_arg1.id)
        lingo_em_arg2 = lingo_doc.get_event_with_id(serif_em_arg2.id)
        found_sent = False

        relation_type = serif_eerm.relation_type
        pattern = serif_eerm.pattern

        for sentence in lingo_doc.sentences:
            if lingo_em_arg1.overlaps_with_anchor(sentence) and lingo_em_arg2.overlaps_with_anchor(sentence):
                return {'text': sentence.text, 'h': {'pos': (lingo_em_arg1.anchors[0].int_pair.first - sentence.start_char_offset(), lingo_em_arg1.anchors[0].int_pair.second - sentence.start_char_offset())},
                        't': {'pos': (lingo_em_arg2.anchors[0].int_pair.first - sentence.start_char_offset(), lingo_em_arg2.anchors[0].int_pair.second - sentence.start_char_offset())},
                        'relation' : relation_type, 'pattern' : pattern}

        if not found_sent:
            raise Exception('No sentence found for serif_eerm.')