import json, re, copy, os, random, pickle
from multiprocessing import Pool
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.local_constants import ARGUMENT_TAG_MAP
import torch
import torch.nn as nn
from nlplingo.oregon.event_models.uoregon.tools.torch_utils import *
from nlplingo.oregon.event_models.uoregon.tools.gcn_utils import *


def get_event_and_span_set(entry):
    event_list = entry['annotation-sets']['abstract-events']['events']
    span_set = entry['annotation-sets']['abstract-events']['span-sets']
    return event_list, span_set


def get_span_text(span_id, span_set):
    text = []
    span_sublist = span_set[span_id]['spans']

    for element in span_sublist:
        text.append(element['string'])
    # assert len(text) == 1, text  # real data could have len(text) > 1 !!!
    assert len(text) > 0
    return text


def remove_slash(object_id):
    return object_id.replace('/', '|')


def get_files_in_dir(dir_path, extension=''):
    if len(extension) > 0:
        file_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if
                      os.path.isfile(os.path.join(dir_path, fname)) and extension == fname[-len(extension):]]
    else:
        file_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if
                      os.path.isfile(os.path.join(dir_path, fname))]

    return file_paths


def get_files_in_dir_recur(dir_path):
    if os.path.isfile(dir_path):
        return [dir_path]
    elif os.path.isdir(dir_path):
        file_list = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if
                     os.path.isfile(os.path.join(dir_path, fname))]
        subdirs = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if
                   os.path.isdir(os.path.join(dir_path, fname))]
        for subdir in subdirs:
            file_list += get_files_in_dir_recur(subdir)
        return file_list
    else:
        return []


def get_subdirs_in_dir(dir_path):
    subdir_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if
                    os.path.isdir(os.path.join(dir_path, fname))]
    return subdir_paths


def read_json(read_path):
    with open(read_path) as f:
        data = json.load(f)
    return data


def get_data_dir(data_name):
    data_name = data_name.strip()
    if re.match(r'^fold\d{,2}$', data_name):
        data_dir = 'cross-valid/' + data_name
    elif data_name in ['en-en', 'en-ar']:
        data_dir = 'simulated/' + data_name
    else:
        data_dir = 'unk'
    return data_dir


def write_json(data, write_path):
    with open(write_path, 'w') as f:
        json.dump(data, f)


def shuffle_list(d_list, random_seed=1996):
    random.seed(random_seed)
    indexs = list(range(len(d_list)))
    for _ in range(10):
        random.shuffle(indexs)
    shuffled = [d_list[new_id] for new_id in indexs]
    return shuffled


def parallelize_list_func(func, inputs):
    '''
    parallelize computattions of a function that takes a list of inputs as arguments and returns a list of outputs
    '''
    num_processors = 20

    chunks = [inputs[i::num_processors] for i in range(num_processors)]

    pool = Pool(processes=num_processors)

    results = pool.map(func, chunks)
    pool.terminate()
    merged = []
    for result in results:
        merged += result

    return merged


def get_features(sentence):
    raise NotImplementedError


def parse_float_list(input_):
    if input_ == None:
        return []
    return list(map(float, input_.split(',')))


def parse_int_list(input_):
    if input_ == None:
        return []
    return list(map(int, input_.split(',')))


def remove_punctuations(string):
    pattern = r'''["’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]+'''
    return re.sub(pattern, '', string)


def get_spans(subtext, text, stanford_tokens):
    spans = []
    for k in range(len(stanford_tokens['word'])):
        if subtext == stanford_tokens['word'][k]:
            spans.append(stanford_tokens['span'][k])
    if len(spans) > 0:
        return spans
    else:
        signal = 0
        if ' {} '.format(subtext) in text:
            signal = 1
            subtext = ' {} '.format(subtext)

        ori_text = copy.deepcopy(text)

        offset = 0
        spans = []
        while len(text) > 0:
            try:
                start_char = text.index(subtext)
                text = text[start_char + len(subtext):]

                start_char += offset
                end_char = start_char + len(subtext) - 1

                assert subtext == ori_text[start_char: end_char + 1]

                offset = end_char + 1

                if signal == 1:
                    spans.append((start_char + 1, end_char - 1))
                else:
                    spans.append((start_char, end_char))

            except ValueError:
                break
        return spans


def get_occurrences(entities, stanford_tokens, text):
    ori_text = copy.deepcopy(text)
    entity_toks = []
    for entity in entities:
        if len(entity.strip()) == 0:
            continue
        spans = get_spans(subtext=entity, text=text, stanford_tokens=stanford_tokens)
        # YS: list[(start-char, end-char)]
        for span in spans:
            start_e, end_e = span
            e_span_set = set(list(range(start_e, end_e + 1)))   # YS: e.g. (3,5) -> set([3,4,5])
            e_toks = []
            for wid in range(len(stanford_tokens['word'])):
                start_w, end_w = stanford_tokens['span'][wid]
                w_span_set = set(list(range(start_w, end_w + 1)))
                if not e_span_set.isdisjoint(w_span_set):
                    e_toks.append(wid)
            # restored = ori_text[stanford_tokens['span'][e_toks[0]][0]: stanford_tokens['span'][e_toks[-1]][1] + 1]
            #
            # if remove_punctuations(restored) != remove_punctuations(entity):
            #     with open('datasets/debug.txt', 'a') as f:
            #         f.write('{}\n{}\n{}\n{}\n{}'.format(
            #             entity, restored, e_toks, stanford_tokens, ori_text
            #         ))
            #         f.write('\n**************************\n')

            entity_toks.append(e_toks)
    return entity_toks


def map_wordpiece_indices(bert_tokenizer, ori_tokens, bert_tokens):
    bert_in_indices = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
    bert_out_indices = []
    ori_index = 0
    bert_index = 0
    while ori_index < len(ori_tokens) and bert_index < len(bert_tokens):
        bert_token = bert_tokens[bert_index]
        if bert_token.startswith('##') or bert_token.startswith(CLS_TOKEN) or bert_token.startswith(SEP_TOKEN):
            bert_index += 1
        else:
            bert_out_indices.append(bert_index)
            ori_index += 1
            bert_index += 1

    if len(bert_out_indices) != len(ori_tokens):
        bert_tokens = ['[CLS]'] + ori_tokens + ['[SEP]']
        bert_in_indices = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_out_indices = list(range(len(bert_in_indices)))[1:-1]

    assert len(bert_out_indices) == len(ori_tokens)
    return bert_in_indices, bert_out_indices


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices.
    unsorted_all[0]: length of each element in batch
    unsorted_all[1]: running index to keep track
    unsorted_all[2]: elements in batch
    unsorted_all:                                   [[5, 4, 2, 3], [0, 1, 2, 3], ['a', 'b', 'c', 'd']]

    zip(*unsorted_all):                             [(5, 0, 'a'), (4, 1, 'b'), (2, 2, 'c'), (3, 3, 'd')]
    sorted(zip(*unsorted_all), reverse=True):       [(5, 0, 'a'), (4, 1, 'b'), (3, 3, 'd'), (2, 2, 'c')]
    zip(*sorted(zip(*unsorted_all), reverse=True)): [(5, 4, 3, 2), (0, 1, 3, 2), ('a', 'b', 'd', 'c')]
    """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def do_padding(tokens_list, batch_size, pad_id=None):
    """ Convert list of list of tokens to a padded LongTensor.
    pad with PAD_ID (= 0), and token_len = max len in the batch
    """
    token_len = max(len(x) for x in tokens_list)
    if pad_id:
        tokens = torch.LongTensor(batch_size, token_len).fill_(pad_id)
    else:
        tokens = torch.LongTensor(batch_size, token_len).fill_(PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def do_padding3d(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len, token_len).fill_(PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s), :len(s)] = torch.LongTensor(s)
        if len(s) < token_len:
            tokens[i, len(s):, len(s):] = 1
    return tokens


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def compute_batch_loss(logits, labels, mask, instance_weights=None):
    '''
    :param logits: [batch size, seq len, num classes]
    :param labels: [batch size, seq len]
    :param mask: [batch size, seq len] => mask out padding tokens in each example
    :return:
    '''
    probs = torch.softmax(logits, dim=2)  # [batch size, seq len, num classes]
    preds = torch.argmax(logits, dim=2)  # [batch size, seq len]
    logits = logits.transpose(1, 2)  # [batch size, num classes, seq len]
    batch_loss = nn.functional.cross_entropy(input=logits, target=labels, reduction='none')  # [batch size, seq len]
    if instance_weights is not None:
        batch_loss = batch_loss * instance_weights.unsqueeze(1)
    masked_loss = batch_loss * mask
    lens = torch.sum(mask, dim=1, keepdim=True)  # [batch size, 1]
    loss = torch.sum(masked_loss) / torch.sum(lens)  # [batch size, 1]

    return loss, probs, preds


def compute_loss(logits, labels):
    '''
    :param logits: [batch size, num classes]
    :param labels: [batch size, ]
    '''
    probs = torch.softmax(logits, dim=1)  # [batch size, num classes]
    preds = torch.argmax(logits, dim=1)  # [batch size,]

    loss = nn.functional.cross_entropy(input=logits, target=labels,
                                       reduction='mean')

    return loss, probs, preds


def is_bad_example(example):
    text = example['text']
    pt1 = r'^\d{4}-\d{2}-\d{2}.+[:]\d{2}:\d{2}[ ]*$'  # time
    pt2 = r'^[A-Z ]+$'  # title
    if re.match(pt1, text) or re.match(pt2, text):
        return True
    elif len(example['word']) <= 2:
        return True
    else:
        return False


def get_better_format_data(data_path):
    data = read_json(read_path=data_path)
    entries = data['entries']
    entry_id_list = list(data['entries'].keys())
    all_examples = []

    for entry_id in entry_id_list:
        entry = entries[entry_id]
        all_examples.append((entry_id, entry))
    return all_examples


def inspect_data(json_file):
    # ************ REAL better data ************
    entries = get_better_format_data(json_file)
    num_sentences = len(entries)
    num_events = 0
    num_spans = 0
    num_string = 0
    max_agents, max_patients = 0, 0
    avg_agents, avg_patients = 0, 0
    max_agents_event, max_patients_event = {}, {}

    max_event_in_sentence = 0
    max_anchors_per_event = 0
    max_anchors_per_event_entry = {}
    avg_anchors_per_event = 0
    num_events_with_multiple_anchors = 0

    max_strings_per_span = 0
    max_strings_per_span_list = []
    avg_strings_per_span = 0

    max_words_per_string = 0
    max_words_per_string_string = ''
    avg_words_per_string = 0

    min_words_per_string = 0
    min_words_per_string_string = ''

    entryid_set = defaultdict(int)
    eventid_set = defaultdict(int)
    spanid_set = defaultdict(int)
    impact_labels = defaultdict(int)
    effect_labels = defaultdict(int)

    for entry_id, entry in entries:
        event_set, span_set = get_event_and_span_set(entry=entry)
        if len(event_set) > max_event_in_sentence:
            max_event_in_sentence = len(event_set)

        entryid_set[entry_id] += 1
        num_events += len(event_set)

        num_spans += len(span_set)
        if len(event_set) > 0:
            for event_id in event_set:
                eventid_set[event_id] += 1
                spanid_set[event_set[event_id]['anchors']] += 1

                trigger_ws = get_span_text(event_set[event_id]['anchors'], span_set)
                avg_anchors_per_event += len(trigger_ws)
                if len(trigger_ws) > max_anchors_per_event:
                    max_anchors_per_event = len(trigger_ws)
                    max_anchors_per_event_entry = entry
                if len(trigger_ws) > 1:
                    num_events_with_multiple_anchors += 1

                if len(trigger_ws) > max_strings_per_span:
                    max_strings_per_span = len(trigger_ws)
                    max_strings_per_span_list = trigger_ws

                agents = []
                for span_id in event_set[event_id]['agents']:
                    spanid_set[span_id] += 1
                    agent_ws = get_span_text(span_id, span_set)
                    agents += agent_ws
                    if len(agent_ws) > max_strings_per_span:
                        max_strings_per_span = len(agent_ws)
                        max_strings_per_span_list = agent_ws

                patients = []
                for span_id in event_set[event_id]['patients']:
                    spanid_set[span_id] += 1
                    patient_ws = get_span_text(span_id, span_set)
                    patients += patient_ws
                    if len(patient_ws) > max_strings_per_span:
                        max_strings_per_span = len(patient_ws)
                        max_strings_per_span_list = patient_ws

                if len(agents) > max_agents:
                    max_agents = len(agents)
                    max_agents_event = entry
                if len(patients) > max_patients:
                    max_patients = len(patients)
                    max_patients_event = entry
                avg_agents += len(agents)
                avg_patients += len(patients)
                num_string += len(agents + patients + trigger_ws)
                for string in agents + patients + trigger_ws:
                    if len(string.split()) > max_words_per_string:
                        max_words_per_string = len(string.split())
                        max_words_per_string_string = string
                for string in agents + patients + trigger_ws:
                    if len(string.split()) < min_words_per_string:
                        min_words_per_string = len(string.split())
                        min_words_per_string_string = string
                impact_labels[event_set[event_id]['helpful-harmful']] += 1
                effect_labels[event_set[event_id]['material-verbal']] += 1
    print('*' * 100)
    print('json file: {}'.format(json_file))
    print('num sentences: {}'.format(num_sentences))
    print('num events: {}'.format(num_events))

    print('max events per sentence: {}'.format(max_event_in_sentence))
    print('avg events per sentence: {}'.format(num_events * 1. / num_sentences))
    print('% events with more than 2 anchors: {} %'.format(num_events_with_multiple_anchors * 100. / num_events))

    print('max anchors per event: {}'.format(max_anchors_per_event))
    print('max anchors per event, that entry: {}'.format(max_anchors_per_event_entry))
    print('avg anchors per event: {}'.format(avg_anchors_per_event * 1. / num_events))

    print('max agents per event: {}'.format(max_agents))
    print('max agents per event, that entry: {}'.format(max_agents_event))
    print('avg agents per event: {}'.format(avg_agents * 1. / num_events))

    print('max patients per event: {}'.format(max_patients))
    print('max patients per event, that entry: {}'.format(max_patients_event))
    print('avg patients per event: {}'.format(avg_patients * 1. / num_events))

    print('max strings per spanid: {}'.format(max_strings_per_span))
    print('max strings per spanid, that list: {}'.format(max_strings_per_span_list))
    print('avg strings per spanid: {}'.format(avg_strings_per_span * 1. / num_spans))

    print('max words per string: {}'.format(max_words_per_string))
    print('max word per string, that string: {}'.format(max_words_per_string_string))
    print('avg words per string: {}'.format(avg_words_per_string * 1. / num_string))

    print('min words per string: {}'.format(min_words_per_string))
    print('min words per string, that string: {}'.format(min_words_per_string_string))
    # print('entry-id set: {}'.format(entryid_set.keys()))
    for entryid in entryid_set:
        assert entryid_set[entryid] == 1
    print('event-id set: {}'.format(eventid_set.keys()))
    print('span-id set: {}'.format(spanid_set.keys()))
    print('impact labels: {}'.format(impact_labels.keys()))
    print('effect labels: {}'.format(effect_labels.keys()))


def permutate(perm_ids, dlist):
    ''' perm_ids[k] = M -> place M-th object in original list to k-th position in new list '''
    assert len(perm_ids) == len(dlist)
    new_list = []
    for perm_id in perm_ids:
        new_list.append(dlist[perm_id])

    return new_list


def inverse_permutate(perm_ids, dlist):
    ''' perm_ids[k] = M -> put k-th object in permutated list -> M-th object in ori list'''
    assert len(perm_ids) == len(dlist)
    new_list = [0 for _ in range(len(dlist))]  # create an empty list
    for k in range(len(perm_ids)):
        new_list[perm_ids[k]] = dlist[k]
    return new_list


def get_ori_idxs(perm_ids, reldist_idxs):
    ori_idxs = [perm_ids[idx] for idx in reldist_idxs]
    return ori_idxs


def get_argument_ids(id2tag, tag_ids, actual_length):
    tag_ids = tag_ids[: actual_length]
    tags = [id2tag[tag_id] for tag_id in tag_ids]
    agent_tags = []
    patient_tags = []
    for tag in tags:
        if 'AGENT' not in tag and 'PATIENT' not in tag:
            agent_tags.append('O')
            patient_tags.append('O')
        else:
            sub_tags = tag.split('|')
            if len(sub_tags) == 1:
                if 'AGENT' in sub_tags[0]:
                    agent_tags.append(sub_tags[0])
                    patient_tags.append('O')
                else:
                    agent_tags.append('O')
                    patient_tags.append(sub_tags[0])
            else:
                agent_tags.append(sub_tags[0])
                patient_tags.append(sub_tags[1])
    assert len(agent_tags) == len(patient_tags)
    agents = []
    patients = []
    agent_toks = []
    patient_toks = []
    for k in range(actual_length):
        if len(agent_toks) > 0 and (agent_tags[k] == 'O' or agent_tags[k].startswith('B-')):
            agents.append(agent_toks)
            agent_toks = []

        if 'AGENT' in agent_tags[k]:
            agent_toks.append(k)

        if len(patient_toks) > 0 and (patient_tags[k] == 'O' or patient_tags[k].startswith('B-')):
            patients.append(patient_toks)
            patient_toks = []

        if 'PATIENT' in patient_tags[k]:
            patient_toks.append(k)

    if len(agent_toks) > 0:
        agents.append(agent_toks)

    if len(patient_toks) > 0:
        patients.append(patient_toks)
    return agents, patients


def get_start_end(ori_ids, span_dict):
    start = 1000 ** 2
    end = - 1000 ** 2
    for ori_id in ori_ids:
        span = span_dict[ori_id]
        start_pos = min(span[0], span[1])
        end_pos = max(span[0], span[1])

        if start_pos < start:
            start = start_pos

        if end_pos > end:
            end = end_pos
    return start, end


def construct_arguments(perm_ids, tag_ids, id2tag, ori_example):
    ori_text = ori_example['text']
    actual_length = len(ori_example['word'])
    tag_ids = tag_ids[: actual_length]
    tags = [id2tag[tag_id] for tag_id in tag_ids]
    agent_tags = []
    patient_tags = []
    for tag in tags:
        if 'AGENT' not in tag and 'PATIENT' not in tag:
            agent_tags.append('O')
            patient_tags.append('O')
        else:
            sub_tags = tag.split('|')
            if len(sub_tags) == 1:
                if 'AGENT' in sub_tags[0]:
                    agent_tags.append(sub_tags[0])
                    patient_tags.append('O')
                else:
                    agent_tags.append('O')
                    patient_tags.append(sub_tags[0])
            else:
                agent_tags.append(sub_tags[0])
                patient_tags.append(sub_tags[1])
    assert len(agent_tags) == len(patient_tags)
    agents = []
    patients = []
    agents_offsets = {}
    patients_offsets = {}

    agent_toks = []
    patient_toks = []
    for k in range(actual_length):
        if len(agent_toks) > 0 and (agent_tags[k] == 'O' or agent_tags[k].startswith('B-')):
            ori_ids = get_ori_idxs(perm_ids, agent_toks)

            start, end = get_start_end(ori_ids, ori_example['span'])

            text_span = ori_text[start: end + 1]
            agents.append(text_span)
            agents_offsets[text_span] = [start, end + 1]
            agent_toks = []

        if 'AGENT' in agent_tags[k]:
            agent_toks.append(k)

        if len(patient_toks) > 0 and (patient_tags[k] == 'O' or patient_tags[k].startswith('B-')):
            ori_ids = get_ori_idxs(perm_ids, patient_toks)

            start, end = get_start_end(ori_ids, ori_example['span'])

            text_span = ori_text[start: end + 1]
            patients.append(text_span)
            patients_offsets[text_span] = [start, end + 1]
            patient_toks = []

        if 'PATIENT' in patient_tags[k]:
            patient_toks.append(k)

    if len(agent_toks) > 0:
        ori_ids = get_ori_idxs(perm_ids, agent_toks)

        start, end = get_start_end(ori_ids, ori_example['span'])

        text_span = ori_text[start: end + 1]
        agents.append(text_span)
        agents_offsets[text_span] = [start, end + 1]

    if len(patient_toks) > 0:
        ori_ids = get_ori_idxs(perm_ids, patient_toks)

        start, end = get_start_end(ori_ids, ori_example['span'])

        text_span = ori_text[start: end + 1]
        patients.append(text_span)
        patients_offsets[text_span] = [start, end + 1]

    return agents, patients, agents_offsets, patients_offsets


def read_pickle(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, write_path):
    with open(write_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def read_aligned_w2v(english_fpath, arabic_fpath):
    def get_w2v(file_path):
        w2v = {}
        with open(file_path) as f:
            lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0][1:]
        for line in lines:
            word = ' '.join(line.split()[:-300]).strip()
            vec = ' '.join(line.split()[-300:]).strip()
            w2v[word] = vec
        return w2v

    bivec_w2v = {}
    eng_w2v = get_w2v(english_fpath)
    ara_w2v = get_w2v(arabic_fpath)
    bivec_w2v.update(eng_w2v)
    bivec_w2v.update(ara_w2v)
    return bivec_w2v


def parse_tsv_report(report_file):
    with open(report_file) as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0][1:-1]
    score_map = {}
    for line in lines:
        feats = line.split('\t')
        assert len(feats) == 15
        entry_id = feats[0]
        score = {
            'f1-event': float(feats[7]),
            'f1-argument': float(feats[-2]),
            'f1-combined': float(feats[-1])
        }
        score_map[entry_id] = score
    return score_map


def get_tokengroups(string_list):
    if len(string_list) == 0:
        return []
    else:
        try:
            return [int(re.findall(r'\d+', string)[0]) for string in string_list]
        except IndexError:
            return []


def read_txt_by_lines(fpath):
    with open(fpath) as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return lines


def split_sentence(text):
    splitters = re.finditer(r'([.?!]+ )[A-Z]+?', text)
    sentences = []
    start = 0
    for splitter in splitters:
        sentence = text[start: int(splitter.start()) + 1]
        sentences.append(sentence)
        start = splitter.end() - 1
    sentence = text[start:]
    sentences.append(sentence)
    return sentences, len(sentences)


def is_equal(str_a, str_b):
    if str_a == str_b:
        return True
    elif len(set(str_a).difference(set(str_b))) == 1:
        return True
    else:
        return False


def align_char_2_token_ENG(tokenized_words, ori_sentence):
    def get_startchar_idx(word, text):
        # ******* search for first non-space character *******
        start_char_idx = 0
        for k in range(len(text)):
            if len(text[k].strip()) > 0:
                start_char_idx = k
                break
        text = text[start_char_idx + len(word):]
        return text, start_char_idx

    char2tok_map = {}
    wrd2span_map = {}
    tok2chars_map = defaultdict(list)
    sent_text = copy.deepcopy(ori_sentence)
    offset = 0

    if len(''.join(tokenized_words)) != len(ori_sentence.replace(' ', '')):
        return None, None, None

    bad_sentence = False
    for w_id, word in enumerate(tokenized_words):
        sent_text, start_char_idx = get_startchar_idx(word, sent_text)
        start_char_idx += offset
        end_char_idx = start_char_idx + len(word) - 1
        offset = end_char_idx + 1

        wrd2span_map[word] = (start_char_idx, end_char_idx)
        if ori_sentence[start_char_idx: end_char_idx + 1] != word:
            bad_sentence = True
            break

        for char_idx in range(start_char_idx, end_char_idx + 1):
            char2tok_map[
                char_idx] = w_id + 1  # 1-indexed style. That means, we always have 1, 2, 3, ..., not 0, 1, 2, ...
            tok2chars_map[w_id + 1].append(char_idx)
    if bad_sentence:
        return None, None, None
    else:
        return char2tok_map, tok2chars_map, wrd2span_map


def is_punct(string):
    if len(re.sub(r'''["’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※]''', '',
                  string)) == 0:
        return True
    else:
        return False


def is_ending_char(char):
    if len(re.findall(
            r'''[ "’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※]''',
            char)) > 0:
        return True
    else:
        return False


def align_char_2_token_ARB(tokenized_words, ori_sentence):
    ori_sentence = ori_sentence[1:]

    def get_start_end(w_id, tokenized_words, sent_text, offset):
        if is_punct(tokenized_words[w_id]):
            ########## current word is an independent word #########
            start_char_idx = -1
            for k in range(len(sent_text)):
                if len(sent_text[k].strip()) > 0:
                    start_char_idx = k
                    break
            end_char_idx = start_char_idx + len(tokenized_words[w_id]) - 1

            start_char_idx += offset
            end_char_idx += offset

            # print('-' * 50 + ' PUNCT')
            # print(sent_text)
            # # import string
            #
            # # tokenized_words[w_id] = [x for x in tokenized_words[w_id] if x in string.printable]
            # print('---{}---'.format(tokenized_words[w_id]))
            # print('start: {}, end: {}'.format(start_char_idx, end_char_idx))
            # print('aligned to: {}'.format(ori_sentence[start_char_idx: end_char_idx + 1]))
            return start_char_idx, end_char_idx, 0
        elif w_id == len(tokenized_words) - 1 or w_id < len(
                tokenized_words) - 1 and '+' != tokenized_words[w_id][-1] and '+' != tokenized_words[w_id + 1][0]:
            ########## current word is an independent word #########
            start_char_idx = -1
            end_char_idx = -1
            for k in range(len(sent_text)):
                if len(sent_text[k].strip()) > 0 and not is_ending_char(sent_text[k]):
                    if start_char_idx < 0:
                        start_char_idx = k
                elif start_char_idx >= 0 and is_ending_char(sent_text[k]):
                    end_char_idx = k - 1
                    break

            start_char_idx += offset
            end_char_idx += offset

            # print('-' * 50 + ' NORMAL')
            # print(sent_text)
            # # import string
            #
            # # tokenized_words[w_id] = [x for x in tokenized_words[w_id] if x in string.printable]
            # print('---{}---'.format(tokenized_words[w_id]))
            # print('start: {}, end: {}'.format(start_char_idx, end_char_idx))
            # print('aligned to: {}'.format(ori_sentence[start_char_idx: end_char_idx + 1]))
            return start_char_idx, end_char_idx, 0
        elif tokenized_words[w_id][-1] == '+':
            ########## current word is the first part of the next word #########
            start_char_idx = -1
            end_char_idx = -1
            for k in range(len(sent_text)):
                if len(sent_text[k].strip()) > 0 and not is_ending_char(sent_text[k]):
                    if start_char_idx < 0:
                        start_char_idx = k
                elif start_char_idx >= 0 and is_ending_char(sent_text[k]):
                    end_char_idx = k - 1
                    break

            start_char_idx += offset
            end_char_idx += offset

            # print('-' * 50 + ' FIRST PART')
            # print(sent_text)
            # print('---', tokenized_words[w_id], tokenized_words[w_id + 1], '---')
            # print('start: {}, end: {}'.format(start_char_idx, end_char_idx))
            # print('aligned to: {}'.format(ori_sentence[start_char_idx: end_char_idx + 1]))
            return start_char_idx, end_char_idx, 1
        else:
            assert tokenized_words[w_id + 1][0] == '+', '0: {}----\n\n\n -1: {}'.format(tokenized_words[w_id + 1][0],
                                                                                        tokenized_words[w_id + 1][-1])
            ########## next word is the second part of the current word #########
            start_char_idx = -1
            end_char_idx = -1
            for k in range(len(sent_text)):
                if len(sent_text[k].strip()) > 0 and not is_ending_char(sent_text[k]):
                    if start_char_idx < 0:
                        start_char_idx = k
                elif start_char_idx >= 0 and is_ending_char(sent_text[k]):
                    end_char_idx = k - 1
                    break

            start_char_idx += offset
            end_char_idx += offset

            # print('-' * 50 + ' SECOND PART')
            # print(sent_text)
            # print('---', tokenized_words[w_id], tokenized_words[w_id + 1], '---')
            # print('start: {}, end: {}'.format(start_char_idx, end_char_idx))
            # print('aligned to: {}'.format(ori_sentence[start_char_idx: end_char_idx + 1]))
            return start_char_idx, end_char_idx, 1

    char2tok_map = {}
    wrd2span_map = {}
    tok2chars_map = defaultdict(list)

    offset = 0
    w_id = 0

    while w_id < len(tokenized_words):
        if tokenized_words[w_id] != '+ه':
            if w_id == len(tokenized_words) - 1:
                start_char_idx = offset
                end_char_idx = len(ori_sentence) - 1
                delta = 0
            else:
                sent_text = ori_sentence[offset:]
                start_char_idx, end_char_idx, delta = get_start_end(w_id, tokenized_words, sent_text, offset)

            offset = end_char_idx + 1

            for char_idx in range(start_char_idx, end_char_idx + 1):
                char2tok_map[
                    char_idx] = w_id + 1  # 1-indexed style. That means, we always have 1, 2, 3, ..., not 0, 1, 2, ...
                tok2chars_map[w_id + 1].append(char_idx)
        else:
            delta = 0
        w_id += 1 + delta
    return ori_sentence, char2tok_map, tok2chars_map, wrd2span_map


def is_bad_sentence(text):
    segments, num_segments = split_sentence(text)
    if num_segments > 1:
        return True
    else:
        is_a_code = True if len(re.findall(r' ', text)) == 0 else False
        is_a_date = True if len(re.findall(r'Year.+Number|Year.+Issue|Volume.+Issue|year.+number', text)) > 0 else False
        is_a_title = True if len(segments[0].split()) < 10 else False
        if is_a_code or is_a_date or is_a_title:
            return True
        else:
            return False


def get_annotations(wa_fpath):
    '''
    datasets/word-align-ar-en/p3/data/parallel_word_aligned/wb/translation/raw/      arb-NG-2-76511-10110213.eng.raw
    datasets/word-align-ar-en/p3/data/parallel_word_aligned/wb/translation/tokenized/arb-NG-2-76511-10110213.eng.tkn

    datasets/word-align-ar-en/p3/data/parallel_word_aligned/wb/source/raw/           arb-NG-2-76511-10110213.arb.raw
    datasets/word-align-ar-en/p3/data/parallel_word_aligned/wb/source/tokenized/     arb-NG-2-76511-10110213.arb.tkn
    datasets/word-align-ar-en/p3/data/parallel_word_aligned/wb/WA/                   arb-NG-2-76511-10110213.wa

    :param wa_fpath:
    :return:
    '''
    en_tkn_fpath = wa_fpath.replace('/WA/', '/translation/tokenized/').rstrip('.wa') + '.eng.tkn'
    ar_tkn_fpath = wa_fpath.replace('/WA/', '/source/tokenized/').rstrip('.wa') + '.arb.tkn'

    en_raw_fpath = en_tkn_fpath.replace('/tokenized/', '/raw/').rstrip('.tkn') + '.raw'
    ar_raw_fpath = ar_tkn_fpath.replace('/tokenized', '/raw/').rstrip('.tkn') + '.raw'

    wa_lines = read_txt_by_lines(wa_fpath)

    en_tkn_lines = read_txt_by_lines(en_tkn_fpath)
    ar_tkn_lines = read_txt_by_lines(ar_tkn_fpath)

    en_raw_lines = read_txt_by_lines(en_raw_fpath)
    ar_raw_lines = read_txt_by_lines(ar_raw_fpath)
    if not (len(wa_lines) == len(en_tkn_lines) == len(ar_tkn_lines) == len(en_raw_lines) == len(ar_raw_lines)):
        return None, None, None, None, None, 0, 0
    else:
        ############# map: eng tok -> arb toks #############
        entok_2_arbtoks_maps = []
        good_line_ids = []
        for line_id, line in enumerate(wa_lines):
            if not is_bad_sentence(en_raw_lines[line_id]):
                tokens = line.split()
                num_tokens = len(tokens)
                en_ars_map = defaultdict(list)
                good_sentence = True
                for k, token in enumerate(tokens):
                    matching = token[-5:]
                    if matching == '(COR)':
                        token = token[:-5]
                        ara_wrds = get_tokengroups(token.split('-')[0].split(','))
                        eng_wrds = get_tokengroups(token.split('-')[1].split(','))
                        if len(eng_wrds) > 0:
                            for wrd in eng_wrds:
                                en_ars_map[wrd].extend(ara_wrds)
                        else:
                            good_sentence = False
                            break
                    else:
                        good_sentence = False
                        break
                if good_sentence:
                    good_line_ids.append(line_id)
                    entok_2_arbtoks_maps.append(en_ars_map)
        ############ map: eng char -> eng tok ###########
        bad_lines = set()

        en_char2tok_maps = []
        en_raw_texts = []
        count1 = 0
        for k, line_id in enumerate(good_line_ids):
            eng_toks = en_tkn_lines[line_id].strip().split()
            char2tok_map = align_char_2_token_ENG(tokenized_words=eng_toks, ori_sentence=en_raw_lines[line_id])[0]
            en_char2tok_maps.append(char2tok_map)
            # ******************
            en_raw_texts.append(en_raw_lines[line_id])
            # ******************
            if char2tok_map is None:
                bad_lines.add(k)
                count1 += 1
        ########### map: arb tok -> arb chars ###########
        ar_tok2chars_maps = []
        ar_raw_texts = []
        count2 = 0
        for k, line_id in enumerate(good_line_ids):
            arb_toks = [token.strip() for token in ar_tkn_lines[line_id].split() if len(token.strip()) > 0]
            ar_raw_text, _, tok2chars_map, _ = \
                align_char_2_token_ARB(tokenized_words=arb_toks, ori_sentence=ar_raw_lines[line_id])
            ar_tok2chars_maps.append(tok2chars_map)
            # *****************
            ar_raw_texts.append(ar_raw_text)
            # *****************
            if len(re.findall(r'[a-zA-Z]', ar_raw_text)) > 0:
                bad_lines.add(k)
                count2 += 1

        assert len(entok_2_arbtoks_maps) == len(en_char2tok_maps) == len(ar_tok2chars_maps) == len(en_raw_texts) == len(
            ar_raw_texts)

        en_raw_texts = [text for k, text in enumerate(en_raw_texts) if k not in bad_lines]
        ar_raw_texts = [text for k, text in enumerate(ar_raw_texts) if k not in bad_lines]
        en_char2tok_maps = [map_ for k, map_ in enumerate(en_char2tok_maps) if k not in bad_lines]
        entok_2_arbtoks_maps = [map_ for k, map_ in enumerate(entok_2_arbtoks_maps) if k not in bad_lines]
        ar_tok2chars_maps = [map_ for k, map_ in enumerate(ar_tok2chars_maps) if k not in bad_lines]

        assert len(entok_2_arbtoks_maps) == len(en_char2tok_maps) == len(ar_tok2chars_maps) == len(en_raw_texts) == len(
            ar_raw_texts)
        return en_raw_texts, ar_raw_texts, en_char2tok_maps, entok_2_arbtoks_maps, ar_tok2chars_maps, count1, count2


def get_start_end_ar(start_end_en, wa_annotations):
    start_en, end_en = start_end_en[0], start_end_en[1]
    en_toks = set()
    for char in range(start_en, end_en + 1):
        if str(char) in wa_annotations['en-char2tok']:
            en_toks.add(wa_annotations['en-char2tok'][str(char)])
    if len(en_toks) == 0:
        return (-1, -1)

    ar_toks = list()
    for en_tok in en_toks:
        if str(en_tok) in wa_annotations['en-ar-tok2toks']:
            ar_toks.extend(wa_annotations['en-ar-tok2toks'][str(en_tok)])

    ar_toks = set(ar_toks)

    if len(ar_toks) == 0:
        return (-1, -1)

    ar_chars = list()
    for ar_tok in ar_toks:
        if str(ar_tok) in wa_annotations['ar-tok2chars']:
            ar_chars.extend(wa_annotations['ar-tok2chars'][str(ar_tok)])

    if len(ar_chars) == 0:
        return (-1, -1)

    start_ar = min(ar_chars)
    end_ar = max(ar_chars)
    return (start_ar, end_ar)


def get_arb_abstract_event(en_abstract_event, wa_annotations):
    def strip_puncts(text):
        return re.sub(
            r'''[^\w\s"’'\(\)\[\]\{\}<>:\,‒–—―!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#%‰‱¶′§~_\|¦⁂☞∴‽※"]''', '',
            text).strip()

    '''
        wa_annotations = {
                            'en-text': en_raw_txts[k],
                            'ar-text': ar_raw_txts[k],
                            'en-char2tok': en_char2tok_maps[k],
                            'en-ar-tok2toks': entok2artoks_maps[k],
                            'ar-tok2chars': artok2chars_maps[k]
                        }
        '''
    anchor_span = en_abstract_event['anchor_offsets'][list(en_abstract_event['anchor_offsets'].keys())[0]]
    agent_spans = en_abstract_event['agent_offsets'][list(en_abstract_event['agent_offsets'].keys())[0]] if len(
        en_abstract_event['agent_offsets']) > 0 else []
    patient_spans = en_abstract_event['patient_offsets'][list(en_abstract_event['patient_offsets'].keys())[0]] if len(
        en_abstract_event['patient_offsets']) > 0 else []
    # ****************************
    anchor_word_en = wa_annotations['en-text'][anchor_span[0]: anchor_span[1] + 1]
    agents_en = [wa_annotations['en-text'][span[0]: span[1] + 1] for span in agent_spans]
    patients_en = [wa_annotations['en-text'][span[0]: span[1] + 1] for span in patient_spans]
    # ****************************
    anchor_span_ar = get_start_end_ar(anchor_span, wa_annotations)
    agent_spans_ar = [get_start_end_ar(agent_span, wa_annotations) for agent_span in agent_spans if
                      agent_span[0] + agent_span[1] != -2]
    patient_spans_ar = [get_start_end_ar(patient_span, wa_annotations) for patient_span in patient_spans if
                        patient_span[0] + patient_span[1] != -2]

    if anchor_span_ar[0] + anchor_span_ar[1] == -2 or len(agent_spans_ar) == 0 or len(patient_spans_ar) == 0:
        return None, None, None, None, None, None
    # ***************************
    anchor_word_ar = wa_annotations['ar-text'][anchor_span_ar[0]: anchor_span_ar[1] + 1]
    agents_ar = [wa_annotations['ar-text'][span[0]: span[1] + 1] for span in agent_spans_ar]
    patients_ar = [wa_annotations['ar-text'][span[0]: span[1] + 1] for span in patient_spans_ar]

    assert len(agents_ar) == len(agents_en) and len(patients_ar) == len(patients_en)

    return anchor_word_ar, agents_ar, patients_ar, anchor_word_en, agents_en, patients_en


if __name__ == '__main__':
    text = '''Please take care, people, especially picnickers, as scorpions have started to appear.'''
    tokenized_words = ['Please', 'take', 'care', ',', 'people', ',', 'especially', 'picnickers', ',', 'as', 'scorpions',
                       'have', 'started', 'to', 'appear', '.']
    char2tok, tok2span = align_char_2_token_ENG(tokenized_words, text)
    print(tokenized_words)
    print(char2tok)
    print(tok2span)
