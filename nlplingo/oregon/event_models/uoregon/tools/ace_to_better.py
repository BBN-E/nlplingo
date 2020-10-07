import os, json
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *

random.seed(1996)


# def get_saying_verbs_ar(saying_verbs_en):
#     with open(os.path.join(WORKING_DIR, 'datasets/bidict_ar-en.txt')) as f:
#         biwords = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines() if
#                    len(line.strip())}
#     saying_verbs_ar = set()
#     for s_verb_en in saying_verbs_en:
#         for word_ar in biwords:
#             word_en = biwords[word_ar].lower()
#             if word_en[:len(s_verb_en)] == s_verb_en:
#                 saying_verbs_ar.add(word_ar)
#     return saying_verbs_ar


saying_verbs_all = set()
saying_verbs_en = {'reveal', 'state', 'suppose', 'ask', 'inquire', 'learn', 'yawn', 'argue', 'persuade', 'mention',
                   'comment', 'explain', 'forbid', 'know', 'realise', 'stutter', 'observe', 'promise', 'frown',
                   'demand', 'reply', 'see', 'guess', 'scream', 'guarantee', 'stammer', 'deny', 'fear', 'think',
                   'warn', 'remember', 'decide', 'expect', 'threaten', 'answer', 'yellask', 'cry', 'claim',
                   'discuss',
                   'report', 'advise', 'repeat', 'beg', 'admit', 'agree', 'confirm', 'understand', 'shout', 'boast',
                   'say', 'chuckle', 'feel', 'recommendrequest', 'instruct', 'prefer', 'tell', 'respond',
                   'discover',
                   'forget', 'suggest', 'teach', 'invite', 'command', 'sigh', 'doubt', 'consider', 'splutter',
                   'laugh',
                   'complain', 'remark', 'snarl', 'swear', 'wonder', 'whisper', 'add', 'describe', 'giggle',
                   'estimate', 'propose', 'insist', 'imagine', 'hope', 'announce'}
saying_verbs_all.update(saying_verbs_en)
# saying_verbs_all.update(get_saying_verbs_ar(saying_verbs_en))


def get_better_label(event):
    e_type = event['type']
    e_subtype = event['subtype']
    e_modality = event['modality']
    e_polarity = event['polarity']
    e_genericity = event['genericity']
    e_tense = event['tense']
    impact_on_patient = ''
    effect_on_patient = ''
    roles_as_agent = []
    roles_as_patient = []
    if e_subtype == 'Transport':
        impact_on_patient = NEUTRAL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Artifact']
    elif e_subtype == 'Demonstrate':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_VERBAL_KEY
        roles_as_agent = ['Entity']
    elif e_subtype == 'Attack':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Attacker']
        roles_as_patient = ['Target']
    elif e_subtype == 'Phone-Write':
        impact_on_patient = NEUTRAL_KEY
        effect_on_patient = VERBAL_KEY
        roles_as_agent = ['Entity']
    elif e_subtype == 'Meet':
        impact_on_patient = NEUTRAL_KEY
        effect_on_patient = MATERIAL_VERBAL_KEY
        roles_as_agent = ['Entity']
    elif e_subtype == 'Nominate':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Person']
    elif e_subtype == 'Start-Position':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Entity']
        roles_as_patient = ['Person']
    elif e_subtype == 'End-Position':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Entity']
        roles_as_patient = ['Person']
    elif e_subtype == 'Elect':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Entity']
        roles_as_patient = ['Person']
    elif e_subtype == 'End-Org':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_patient = ['Org']
    elif e_subtype == 'Start-Org':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Org']
    elif e_subtype == 'Merge-Org':
        impact_on_patient = NEUTRAL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_patient = ['Org']
    elif e_subtype == 'Declare-Bankruptcy':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Org']
    elif e_subtype == 'Transfer-Ownership':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Buyer']
        roles_as_patient = ['Beneficiary']
    elif e_subtype == 'Transfer-Money':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Giver']
        roles_as_patient = ['Beneficiary', 'Recipient']
    elif e_subtype == 'Die':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Victim']
    elif e_subtype == 'Marry':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Person']
    elif e_subtype == 'Divorce':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Person']
    elif e_subtype == 'Injure':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Victim']
    elif e_subtype == 'Be-Born':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_patient = ['Person']
    elif e_subtype == 'Charge-Indict':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Prosecutor']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Trial-Hearing':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Prosecutor', 'Adjudicator']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Release-Parole':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Entity']
        roles_as_patient = ['Person']
    elif e_subtype == 'Appeal':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Defendant']
        roles_as_patient = ['Adjudicator']
    elif e_subtype == 'Extradite':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Person']
    elif e_subtype == 'Acquit':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Adjudicator']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Arrest-Jail':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Person']
    elif e_subtype == 'Fine':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Adjudicator']
        roles_as_patient = ['Entity']
    elif e_subtype == 'Convict':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Adjudicator']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Sue':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Plaintiff']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Pardon':
        impact_on_patient = HELPFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Adjudicator']
        roles_as_patient = ['Defendant']
    elif e_subtype == 'Execute':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Agent']
        roles_as_patient = ['Person']
    elif e_subtype == 'Sentence':
        impact_on_patient = HARMFUL_KEY
        effect_on_patient = MATERIAL_KEY
        roles_as_agent = ['Adjudicator']
        roles_as_patient = ['Defendant']

    # if e_polarity == 'Negative':
    #     impact_on_patient = NEUTRAL_KEY

    impact_on_patient, effect_on_patient, set(roles_as_agent), set(roles_as_patient)
    label = {
        'impact': impact_on_patient,
        'effect': effect_on_patient,
        'agents': roles_as_agent,
        'patients': roles_as_patient
    }
    return label


def update_better_label(anchor_ids, sent_id, all_sentences, old_label):
    tokens = all_sentences[sent_id]
    trigger_words = set([tokens[anchor_id]['lemma'].lower() for anchor_id in anchor_ids])
    '''
    saying verbs are taken from this source: https://krauexchange.wordpress.com/2011/10/09/saying-verbs/?fbclid=IwAR0o0ozWhQtt29Znrxrjbka0jMAUr_t5n2LTIxiSiO2SJ49ltLlT8f92avI
    '''

    if len(saying_verbs_all & trigger_words) > 0:
        return VERBAL_KEY
    else:
        return old_label


def get_better_dpoints(ace_doc):
    original_text = ace_doc['original_text']
    all_events = [sentence['labels']['event_mentions'] for sentence in ace_doc['sentences']]
    all_entities = [sentence['labels']['entity_mentions'] for sentence in ace_doc['sentences']]
    all_values = [sentence['labels']['value_mentions'] for sentence in ace_doc['sentences']]
    all_timex2s = [sentence['labels']['timex2_mentions'] for sentence in ace_doc['sentences']]
    all_sentences = [sentence['inputs']['tokens'] for sentence in ace_doc['sentences']]
    better_dpoints = {}
    doc_id = ace_doc['id']
    all_segment_set = set()
    no_event_sentences = []
    event_sentences_without_agent_patient = []
    for sent_id, sentence in enumerate(ace_doc['sentences']):
        event_mention_list = sentence['labels']['event_mentions']

        if len(event_mention_list) > 0:
            first_em = event_mention_list[0]
            previous_segment = (int(first_em['ldc_scope']['start']), int(first_em['ldc_scope']['end']))
            all_segment_set.add(previous_segment)

            added_span_ids = []
            has_agent_patient = False
            segment_id = len(all_segment_set)
            entry_id = 'doc-{}_{}_0'.format(doc_id, segment_id)
            entry_content = {
                'annotation-sets': {
                    'abstract-events': {
                        'events': {},
                        'span-sets': {}
                    }
                },
                'doc-id': 'doc-{}'.format(doc_id),
                'entry-id': entry_id,
                'segment-text': '',
                'segment-type': 'sentence',
                'sent-id': segment_id
            }
            for ev_id, event_info in enumerate(event_mention_list):
                current_segment = (int(event_info['ldc_scope']['start']), int(event_info['ldc_scope']['end']))
                if current_segment != previous_segment:
                    # ******** Finish with previous segment
                    entry_content['segment-text'] = original_text[previous_segment[0]: previous_segment[1] + 1]
                    better_dpoints[entry_id] = entry_content
                    # ******** Start with new segment
                    all_segment_set.add(current_segment)
                    previous_segment = current_segment

                    added_span_ids = []
                    has_agent_patient = False
                    segment_id = len(all_segment_set)
                    entry_id = 'doc-{}_{}_0'.format(doc_id, segment_id)
                    entry_content = {
                        'annotation-sets': {
                            'abstract-events': {
                                'events': {},
                                'span-sets': {}
                            }
                        },
                        'doc-id': 'doc-{}'.format(doc_id),
                        'entry-id': entry_id,
                        'segment-text': '',
                        'segment-type': 'sentence',
                        'sent-id': segment_id
                    }

                event_id = 'event{}'.format(len(entry_content['annotation-sets']['abstract-events']['events']) + 1)

                start_anchor = int(event_info['anchor']['start'])
                end_anchor = int(event_info['anchor']['end'])

                span_set_id = 'ss-{}'.format(len(entry_content['annotation-sets']['abstract-events']['span-sets']) + 1)
                entry_content['annotation-sets']['abstract-events']['span-sets'][span_set_id] = {
                    'spans': [
                        {
                            'string': original_text[start_anchor: end_anchor + 1]
                        }
                    ]
                }
                added_span_ids.append(span_set_id)

                label = get_better_label(event_info)
                label['effect'] = update_better_label(
                    event_info['anchor']['token_ids'], sent_id, all_sentences, label['effect']
                )

                event_content = {
                    'agents': [],
                    'patients': [],
                    # 'agent_wtypes': [],
                    # 'patient_wtypes': [],
                    'anchors': span_set_id,
                    'eventid': event_id,
                    'helpful-harmful': label['impact'],
                    'material-verbal': label['effect'],
                }

                event_args = event_info['args']
                arg_agents = []
                arg_patients = []
                for arg in event_args:
                    role = arg['role']
                    if role in label['agents']:
                        role_type = 'agents'
                        # word_type = 'agent_wtypes'
                    elif role in label['patients']:
                        role_type = 'patients'
                        # word_type = 'patient_wtypes'
                    else:
                        continue

                    location = arg['loc']
                    obj_type, obj_sent_id, obj_id = location

                    if obj_type == 'entity':
                        # argument_object = all_entities[obj_sent_id][obj_id]['head']
                        argument_object = all_entities[obj_sent_id][obj_id]['extent']
                        # argument_wtype = all_entities[obj_sent_id][obj_id]['word_type']
                    elif obj_type == 'value':
                        argument_object = all_values[obj_sent_id][obj_id]['extent']
                        # argument_wtype = 'VALUE'
                    elif obj_type == 'timex2':
                        argument_object = all_timex2s[obj_sent_id][obj_id]['extent']
                        # argument_wtype = 'TIME'
                    else:
                        print('UNKNOWN ROLE! {}'.format(arg))
                        continue
                    assert obj_sent_id == sent_id
                    start_obj = int(argument_object['start'])
                    end_obj = int(argument_object['end'])

                    if role_type == 'agents':
                        arg_agents.append((start_obj, end_obj))
                    else:
                        arg_patients.append((start_obj, end_obj))

                if len(arg_agents) > 0:
                    has_agent_patient = True
                    span_set_id = 'ss-{}'.format(
                        len(entry_content['annotation-sets']['abstract-events']['span-sets']) + 1)
                    entry_content['annotation-sets']['abstract-events']['span-sets'][span_set_id] = {
                        'spans': []
                    }
                    event_content['agents'].append(span_set_id)
                    added_span_ids.append(span_set_id)

                    for arg_agent in arg_agents:
                        start_a, end_a = arg_agent
                        text_a = original_text[start_a: end_a + 1]
                        entry_content['annotation-sets']['abstract-events']['span-sets'][span_set_id]['spans'].append(
                            {
                                'string': text_a
                            }
                        )

                if len(arg_patients) > 0:
                    has_agent_patient = True
                    span_set_id = 'ss-{}'.format(
                        len(entry_content['annotation-sets']['abstract-events']['span-sets']) + 1)
                    entry_content['annotation-sets']['abstract-events']['span-sets'][span_set_id] = {
                        'spans': []
                    }
                    event_content['patients'].append(span_set_id)
                    added_span_ids.append(span_set_id)

                    for arg_patient in arg_patients:
                        start_p, end_p = arg_patient
                        text_p = original_text[start_p: end_p + 1]
                        entry_content['annotation-sets']['abstract-events']['span-sets'][span_set_id]['spans'].append(
                            {
                                'string': text_p
                            }
                        )
                if event_content['helpful-harmful'] != HARMFUL_KEY:
                    entry_content['annotation-sets']['abstract-events']['events'][event_id] = event_content
                elif random.random() >= 0.5:
                    entry_content['annotation-sets']['abstract-events']['events'][event_id] = event_content

            if has_agent_patient:
                entry_content['segment-text'] = original_text[previous_segment[0]: previous_segment[1] + 1]
                better_dpoints[entry_id] = entry_content
        else:
            start_sent = int(sentence['inputs']['spans'][0][0])
            end_sent = int(sentence['inputs']['spans'][-1][-1])
            raw_text = original_text[start_sent: end_sent + 1]

            segment_id = len(all_segment_set)
            entry_id = 'doc-{}_{}_0'.format(doc_id, segment_id)
            entry_content = {
                'annotation-sets': {
                    'abstract-events': {
                        'events': {},
                        'span-sets': {}
                    }
                },
                'doc-id': 'doc-{}'.format(doc_id),
                'entry-id': entry_id,
                'segment-text': raw_text,
                'segment-type': 'sentence',
                'sent-id': segment_id
            }
            better_dpoints[entry_id] = entry_content
            all_segment_set.add((start_sent, end_sent))

    return better_dpoints


def generate_data_for_better(language='English'):
    with open(os.path.join('datasets', 'multi-ace-all-features.json')) as f:
        ace = json.load(f)
    BETTER_dataset = {
        "corpus-id": 'ace_2005_td_v7',
        "entries": {},
        "format-type": "bp-corpus",
        "format-version": "v8f",
        "provenance": {
            "annotation-procedure": "",
            "annotation-role": "",
            "annotator-class": "",
            "annotator-id": "",
            "corpus-creation-date": ""
        }
    }
    for ace_doc in ace[language]:
        better_dpoints = get_better_dpoints(ace_doc)
        for example_id in better_dpoints:
            example_content = better_dpoints[example_id]
            BETTER_dataset['entries'][example_id] = example_content

    with open(os.path.join('datasets', 'ace2005.{}.bp.json'.format(language)), 'w') as f:
        json.dump(BETTER_dataset, f)
    return BETTER_dataset


def examine_better_data(data_path=None):
    if data_path is None:
        data_path = os.path.join('datasets/abstract-0d.bp.json')
    data = read_json(read_path=data_path)
    entries = data['entries']
    print(len(entries))
    print_out = []
    for entry_id in entries:
        entry = entries[entry_id]
        segment_text = entry['segment-text']
        segment_type = entry['segment-type']
        events = entry['annotation-sets']['abstract-events']['events']
        span_sets = entry['annotation-sets']['abstract-events']['span-sets']

        multiple_spans = set()
        for span in span_sets:
            span_strs = span_sets[span]
            assert len(span_strs['spans']) > 0  # a span is a group of substrings which are not necessarily adjacent
            if len(span_strs['spans']) > 1:
                multiple_spans.add(span)

            for sp in span_strs['spans']:
                assert len(list(sp.keys())) == 1

        for event_id in events:
            event = events[event_id]
            agents = event['agents']
            patients = event['patients']
            anchors = event['anchors']
            impact = event[IMPACT_KEY]
            effect = event[EFFECT_KEY]

            # if anchors in multiple_spans:
            #     print('-------------------')
            #     print(entry)
            #     print(anchors)

            assert len(anchors.split('-')) == 2
            assert len(agents) <= 1 and len(patients) <= 1

            for agent in agents:
                assert agent in span_sets
            for patient in patients:
                assert patient in span_sets

            assert impact in [HARMFUL_KEY, HELPFUL_KEY, NEUTRAL_KEY, UNKNOWN_EVENT_KEY]
            assert effect in [MATERIAL_KEY, VERBAL_KEY, MATERIAL_VERBAL_KEY, UNKNOWN_EVENT_KEY], effect

            if effect == UNKNOWN_EVENT_KEY:
                # print('--------------------------')
                print_out.append('{}'.format(entry))

    print('[' + ','.join(print_out) + ']')


if __name__ == '__main__':
    # examine_better_data()
    BETTER_eng = generate_data_for_better(language='English')
    BETTER_ara = generate_data_for_better(language='Arabic')
    print('len(BETTER_eng): {}'.format(len(BETTER_eng['entries'])))
    print('len(BETTER_ara): {}'.format(len(BETTER_ara['entries'])))
    '''
    len(BETTER_eng): 16582  entries
    len(BETTER_ara): 3656   entries
    '''