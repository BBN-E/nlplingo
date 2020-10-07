from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.tools.corpus_utils import Corpus, AbstractEvent, Sentence, EmptyCorpus, \
    strip_punctuations
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.local_constants import *
from nlplingo.oregon.event_models.uoregon.define_opt import opt, logger

MODEL_DIR = opt['datapoint_dir']	# ==>

def get_ED_data(bbn_sentences):
    all_data = []

    num_entry = 0
    total_entries = len(bbn_sentences)
    for sentence in bbn_sentences:
        num_entry += 1
        if num_entry % 20 == 0:
            print('processed {}/{}'.format(num_entry, total_entries))

        if len(sentence.features['word']) <= 1:
            continue
        else:
            datapoint = {
                'id': sentence.entry_id,
                'entry_id': sentence.entry_id,
                'triggers': [],
                'event-types': [],
                'text': sentence.text,
                'ori_text': sentence.ori_text,
                'ori_entry': sentence.to_json_dict(),
                'norm2ori_offsetmap': sentence.norm2ori_offsetmap,
                'lang': sentence.lang
            }

            for feat in sentence.features:
                datapoint[feat] = sentence.features[feat]
            for event_id in sentence.abstract_events:
                abs_event = sentence.abstract_events[event_id]
                trigger_w = abs_event.anchors.spans[0].text  # assuming that each event has only one trigger
                trigger_occs = get_occurrences([trigger_w], sentence.features, sentence.text)   # YS, e.g. [[6]]


                if len(trigger_occs) > 0 and len(trigger_occs[0]) > 0:
                    datapoint['triggers'].append(trigger_occs[0])
                    datapoint['event-types'].append(
                        '{}|{}'.format(
                            abs_event.helpful_harmful,
                            abs_event.material_verbal
                        )
                    )
            if len(sentence.abstract_events) > 0 and len(datapoint['triggers']) == 0:  # remove 'broken' examples
                continue
            all_data.append(datapoint)

    print('ED size: {}'.format(len(all_data)))
    return all_data


def get_argument_data(bbn_sentences):
    all_data = []

    num_entry = 0
    total_entries = len(bbn_sentences)
    for sentence in bbn_sentences:
        num_entry += 1
        if num_entry % 20 == 0:
            print('processed {}/{}'.format(num_entry, total_entries))

        if len(sentence.features['word']) <= 1:
            continue
        else:
            for event_id in sentence.abstract_events:
                abs_event = sentence.abstract_events[event_id]
                trigger_w = abs_event.anchors.spans[0].text  # assuming that each event has only one trigger
                trigger_occs = get_occurrences([trigger_w], sentence.features, sentence.text)
                if len(trigger_occs) > 0 and len(trigger_occs[0]) > 0:
                    agents, patients = [], []
                    for agent in abs_event.agents:
                        for span in agent.spans:
                            agents.append(span.text)
                    for patient in abs_event.patients:
                        for span in patient.spans:
                            patients.append(span.text)

                    datapoint = {
                        'id': '{}_{}'.format(sentence.entry_id, abs_event.event_id),
                        'entry_id': sentence.entry_id,
                        'trigger': trigger_occs[0],
                        'agents': get_occurrences(agents, sentence.features, sentence.text),
                        'patients': get_occurrences(patients, sentence.features, sentence.text),
                        'text': sentence.text,
                        'ori_text': sentence.ori_text,
                        'ori_event': abs_event.to_json_dict(),
                        'ori_entry': sentence.to_json_dict(),
                        'norm2ori_offsetmap': sentence.norm2ori_offsetmap,
                        'lang': sentence.lang
                    }
                    for feat in sentence.features:
                        datapoint[feat] = sentence.features[feat]
                    all_data.append(datapoint)

    print('Argument size: {}'.format(len(all_data)))
    return all_data


def get_pipeline_data(bbn_sentences):
    all_data = []

    num_entry = 0
    total_entries = len(bbn_sentences)
    for sentence in bbn_sentences:
        num_entry += 1
        if num_entry % 20 == 0:
            print('processed {}/{}'.format(num_entry, total_entries))

        if len(sentence.features['word']) > 1:
            datapoint = {
                'id': sentence.entry_id,
                'entry_id': sentence.entry_id,
                'text': sentence.text,
                'ori_text': sentence.ori_text,
                'ori_entry': sentence.to_json_dict(),
                'norm2ori_offsetmap': sentence.norm2ori_offsetmap,
                'lang': sentence.lang
            }
            for feat in sentence.features:
                datapoint[feat] = sentence.features[feat]

            all_data.append(datapoint)

    print('Combined task size: {}'.format(len(all_data)))
    return all_data


def read_abstract_train_data(train_file, dev_file=None):
    data_version = os.path.basename(train_file).split('.')[0]

    # YS: e.g.: python/clever/event_models/uoregon/models/pipeline/_01/data/abstract-8d-inclusive/
    data_dir = os.path.join(MODEL_DIR, 'data/{}'.format(data_version))

    ensure_dir(os.path.join(MODEL_DIR, 'data'))
    ensure_dir(data_dir)

    train_corpus = Corpus(filepath=train_file, lang=opt['input_lang'])

    train_sentences = train_corpus.sentences
    train_sentences = shuffle_list(train_sentences)
    # ************ split data ******************
    if dev_file is None:
        train_size = int(len(train_sentences) * 0.9)
        train_data = train_sentences[:train_size]
        dev_data = train_sentences[train_size:]
    else:
        dev_corpus = Corpus(filepath=dev_file, lang=opt['input_lang'])
        # ******************************************
        train_data = train_sentences
        dev_data = dev_corpus.sentences

    ############ co-training with generated arabic wa corpus ############
    if opt['co_train_lambda'] > 0:
        arb_train_fpath = 'datasets/8d/update2/arabic-wa-corpus{}.bp.json'.format('.removed_incomplete' if opt[
            'remove_incomplete'] else '')
        arb_dev_fpath = 'datasets/8d/update2/arabic-abstract-sample.bp.json'
        arb_train_corpus = Corpus(filepath=arb_train_fpath, lang='arabic')
        arb_dev_corpus = Corpus(filepath=arb_dev_fpath, lang='arabic')

        train_data += arb_train_corpus.sentences
        dev_data += arb_dev_corpus.sentences
        print('[Co-training] Adding {} arabic entries to train data'.format(len(arb_train_corpus.sentences)))
        print('[Co-training] Adding {} arabic entries to dev data'.format(len(arb_dev_corpus.sentences)))
        logger.info('[Co-training] Adding {} arabic entries to train data'.format(len(arb_train_corpus.sentences)))
        logger.info('[Co-training] Adding {} arabic entries to dev data'.format(len(arb_dev_corpus.sentences)))

    prefix = 'data/{}/{}'.format(data_version, data_version)

    write_json(
        data=get_ED_data(train_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.ED.train')
    )
    write_json(
        data=get_argument_data(train_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.argument.train')
    )

    write_json(
        data=get_ED_data(dev_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.ED.dev')
    )
    write_json(
        data=get_argument_data(dev_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.argument.dev')
    )

    data_map = {
        'ED': {
            'train': prefix + '.ED.train',
            'dev': prefix + '.ED.dev'
        },
        'argument': {
            'train': prefix + '.argument.train',
            'dev': prefix + '.argument.dev'
        }
    }
    return data_map


def read_abstract_test_data(test_file):
    data_version = os.path.basename(test_file).split('.')[0]

    ensure_dir(os.path.join(MODEL_DIR, 'data'))
    ensure_dir(os.path.join(MODEL_DIR, 'data/{}'.format(data_version)))

    test_corpus = Corpus(filepath=test_file, lang=opt['input_lang'])

    test_corpus.clear_annotation()

    test_entries = test_corpus.sentences

    prefix = 'data/{}/{}'.format(data_version, data_version)

    write_json(
        data=get_pipeline_data(test_entries),
        write_path=os.path.join(MODEL_DIR, prefix + '.pipeline.test')
    )
    data_map = {
        'pipeline': {
            'test': prefix + '.pipeline.test'
        }
    }
    return data_map


def select_hidden_data_by_strategy(hidden_train_file, sentences, select_strategy):
    '''
    For now, add all hidden data to existing training data
    '''
    if select_strategy.endswith('all'):
        return sentences
    elif select_strategy.endswith('bad'):
        report_file = os.path.join('hideval', '{}.tsv'.format(os.path.basename(hidden_train_file)))
        assert os.path.exists(report_file)
        score_map = parse_tsv_report(report_file)
        selected_sentences = []
        for sent in sentences:
            entry_id = sent.entry_id
            if entry_id in score_map:
                f1_event, f1_arg, f1_comb = score_map[entry_id]['f1-event'], score_map[entry_id]['f1-argument'], \
                                            score_map[entry_id]['f1-combined']
                if f1_comb < opt['bad_threshold']:
                    selected_sentences.append(sent)
        return selected_sentences
    else:
        return []


def read_abstract_train_data_from_files(train_files, dev_file=None, observed_included=True):
    data_version = 'observed+hidden'
    data_dir = os.path.join(MODEL_DIR, 'data/{}'.format(data_version))

    ensure_dir(os.path.join(MODEL_DIR, 'data'))
    ensure_dir(data_dir)

    all_train_sentences = []
    for k, train_file in enumerate(train_files):
        # **************************************
        logger.info('-------------- processing -------------')
        logger.info('Train file: {}'.format(train_file))
        train_corpus = Corpus(filepath=train_file, lang=opt['input_lang'])
        if observed_included and k == 0:  # existing training data:
            all_train_sentences += train_corpus.sentences
            logger.info('-> take all')
        else:
            tmp = len(all_train_sentences)
            all_train_sentences += select_hidden_data_by_strategy(train_file, train_corpus.sentences,
                                                                  opt['train_strategy'])
            logger.info('-> take {}/{}'.format(len(all_train_sentences) - tmp, len(train_corpus.sentences)))
    all_train_sentences = shuffle_list(all_train_sentences)
    # ************ split data ******************
    if dev_file is None:
        train_size = int(len(all_train_sentences) * 0.9)
        train_data = all_train_sentences[:train_size]
        dev_data = all_train_sentences[train_size:]
    else:
        dev_corpus = Corpus(filepath=dev_file, lang=opt['input_lang'])
        # ******************************************
        train_data = all_train_sentences
        dev_data = dev_corpus.sentences

    ############ co-training with generated arabic wa corpus ############
    if opt['co_train_lambda'] > 0:
        arb_train_fpath = 'datasets/8d/update2/arabic-wa-corpus{}.bp.json'.format('.removed_incomplete' if opt[
            'remove_incomplete'] else '')
        arb_dev_fpath = 'datasets/8d/update2/arabic-abstract-sample.bp.json'
        arb_train_corpus = Corpus(filepath=arb_train_fpath, lang='arabic')
        arb_dev_corpus = Corpus(filepath=arb_dev_fpath, lang='arabic')

        train_data += arb_train_corpus.sentences
        dev_data += arb_dev_corpus.sentences
        print('[Co-training] Adding {} arabic entries to train data'.format(len(arb_train_corpus.sentences)))
        print('[Co-training] Adding {} arabic entries to dev data'.format(len(arb_dev_corpus.sentences)))
        logger.info('[Co-training] Adding {} arabic entries to train data'.format(len(arb_train_corpus.sentences)))
        logger.info('[Co-training] Adding {} arabic entries to dev data'.format(len(arb_dev_corpus.sentences)))

    prefix = 'data/{}/{}'.format(data_version, data_version)

    write_json(
        data=get_ED_data(train_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.ED.train')
    )
    write_json(
        data=get_argument_data(train_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.argument.train')
    )

    write_json(
        data=get_ED_data(dev_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.ED.dev')
    )
    write_json(
        data=get_argument_data(dev_data),
        write_path=os.path.join(MODEL_DIR, prefix + '.argument.dev')
    )

    data_map = {
        'ED': {
            'train': prefix + '.ED.train',
            'dev': prefix + '.ED.dev'
        },
        'argument': {
            'train': prefix + '.argument.train',
            'dev': prefix + '.argument.dev',
        }
    }
    return data_map


def create_observed_hidden_data(train_file):
    train_corpus = Corpus(filepath=train_file, lang='english', parsing=False)
    '''
    self.__lang = lang
        self.__corpus_id = data['corpus-id']
        self.__format_type = data['format-type']
        self.__format_version = data['format-version']
        self.__provenance = data['provenance']
        self.__parsing = parsing
    '''
    init_info = {
        'lang': train_corpus.lang,
        'format-type': train_corpus.format_type,
        'format-version': train_corpus.format_version,
        'provenance': train_corpus.provenance,
    }
    num_docs = len(train_corpus.docs)

    num_observed = int(num_docs / 2)
    observed_sents, hidden_sents = [], []

    all_docs = [(doc_id, doc) for doc_id, doc in train_corpus.docs.items()]
    all_docs = shuffle_list(all_docs)

    count = 0
    for doc_id, doc in all_docs:
        if count < num_observed:
            observed_sents += doc.sentences
        else:
            hidden_sents += doc.sentences
        count += 1

    simulated_dir = 'hideval'
    ensure_dir(simulated_dir)
    ############## observed corpus ##############
    init_info['corpus-id'] = train_corpus.corpus_id + '.observed'
    observed_corpus = EmptyCorpus(init_info)
    observed_corpus.add_sentences(observed_sents)
    observed_corpus.save(os.path.join(simulated_dir, os.path.basename(train_file) + '.observed'))
    ############## hidden corpus ################
    init_info['corpus-id'] = train_corpus.corpus_id + '.hidden'
    observed_corpus = EmptyCorpus(init_info)
    observed_corpus.add_sentences(hidden_sents)
    observed_corpus.save(os.path.join(simulated_dir, os.path.basename(train_file) + '.hidden'))


def create_wa_corpus_eng():
    align_data_dirs = [
        'datasets/word-align-ar-en/p1/data/parallel_word_aligned',
        'datasets/word-align-ar-en/p2/data/parallel_word_aligned',
        'datasets/word-align-ar-en/p3/data/parallel_word_aligned',
        'datasets/word-align-ar-en/p4/data/parallel_word_aligned',
        'datasets/word-align-ar-en/p5/data/parallel_word_aligned',
    ]
    wa_corpus = {
        'corpus-id': 'word-alignment.LDC2014T05.LDC2014T10.LDC2014T14.LDC2014T19.LDC2014T22.eng',
        'entries': {},
        'format-type': 'bp-corpus',
        'format-version': 'v9',
        'provenance': {
            "annotation-procedure": "extract-from-word-alignment-ar-en-corpus",
            "annotation-role": "nothing",
            "annotator-class": "UOregon",
            "annotator-id": "minh",
            "corpus-creation-date": "2020-04-02_19-57"
        }
    }
    num_wa_fpaths = 0
    num_sents = 0
    enfails = 0
    arfails = 0
    for align_dir in align_data_dirs:
        domain_dirs = get_subdirs_in_dir(align_dir)
        for domain_dir in domain_dirs:
            wa_fpaths = [fname for fname in get_files_in_dir(os.path.join(domain_dir, 'WA'))]
            for fpath in wa_fpaths:

                en_raw_txts, ar_raw_txts, en_char2tok_maps, entok2artoks_maps, artok2chars_maps, count1, count2 = get_annotations(
                    wa_fpath=fpath)
                enfails += count1
                arfails += count2
                if en_raw_txts is not None:
                    num_wa_fpaths += 1
                    for k in range(len(en_raw_txts)):
                        wa_annotations = {
                            'en-text': en_raw_txts[k],
                            'ar-text': ar_raw_txts[k],
                            'en-char2tok': en_char2tok_maps[k],
                            'en-ar-tok2toks': entok2artoks_maps[k],
                            'ar-tok2chars': artok2chars_maps[k]
                        }

                        entry_value = {
                            'doc-id': os.path.basename(fpath).rstrip('.wa'),
                            'entry-id': '{}_{}_0'.format(os.path.basename(fpath).rstrip('.wa'), k + 1),
                            'segment-text': wa_annotations['en-text'],
                            'segment-type': 'sentence',
                            'sent-id': str(k + 1),
                            "annotation-sets": {
                                "abstract-events": {}
                            },
                            'wa-annotations': wa_annotations
                        }
                        num_sents += 1
                        wa_corpus['entries'][entry_value['entry-id']] = entry_value
    write_json(
        data=wa_corpus,
        write_path='datasets/8d/update2/english-wa-corpus.bp.json'
    )
    print('num wa fpaths', num_wa_fpaths)
    print('Writing to file: wa corpus: {} entries'.format(len(wa_corpus['entries'])))


def is_good_event(abs_event, event_table, argument_table):
    event_prob = abs_event['event_prob']  # event_prob.shape = [16, ]
    token_probs = abs_event['argument_prob']  # argument_prob.shape = [num words, num BIO tags]

    event_type_id = np.argmax(event_prob)
    condition_1 = False
    condition_2 = True

    if event_table[str(event_type_id)][0] <= event_prob[event_type_id] <= event_table[str(event_type_id)][1]:
        condition_1 = True

    for token_id in range(len(token_probs)):
        bio_id = np.argmax(token_probs[token_id])
        if not (argument_table[str(bio_id)][0] <= token_probs[token_id][bio_id] <= argument_table[str(bio_id)][1]):
            condition_2 = False
            break
    if condition_1 and condition_2:
        return True
    else:
        return False


def load_tables():
    event_table = read_json('datasets/event_table.prob')
    argument_table = read_json('datasets/argument_table.prob')
    ev_table = {}
    arg_table = {}

    for ev_id in event_table:
        if len(event_table[ev_id]) == 0:
            ev_table[ev_id] = (-1, -1)
        else:
            ev_table[ev_id] = (min(event_table[ev_id]), max(event_table[ev_id]))

    for bio_id in argument_table:
        if len(argument_table[bio_id]) == 0:
            arg_table[bio_id] = (-1, -1)
        else:
            arg_table[bio_id] = (min(argument_table[bio_id]), max(argument_table[bio_id]))
    return ev_table, arg_table


def create_wa_corpus_arb():
    annotation_procedure = '''english-wa-corpus -> extract texts -> eval with trained model -> arabic-english word alignments -> arabic-wa-corpus'''
    train_dev_wa_corpus_arb = EmptyCorpus(
        init_info={
            'lang': 'arabic',
            'corpus-id': 'word-alignment.LDC2014T05.LDC2014T10.LDC2014T14.LDC2014T19.LDC2014T22.arb' + '.removed_incomplete' if
            opt[
                'remove_incomplete'] else '',
            'format-type': 'bp-corpus',
            'format-version': 'v9',
            'provenance': {
                "annotation-procedure": annotation_procedure,
                "annotation-role": "",
                "annotator-class": "UOregon",
                "annotator-id": "UOregon",
                "corpus-creation-date": "2020-04-02_19-57"
            }
        }
    )
    train_wa_corpus_arb = EmptyCorpus(
        init_info={
            'lang': 'arabic',
            'corpus-id': 'word-alignment.LDC2014T05.LDC2014T10.LDC2014T14.LDC2014T19.LDC2014T22.arb.train' + '.removed_incomplete' if
            opt[
                'remove_incomplete'] else '',
            'format-type': 'bp-corpus',
            'format-version': 'v9',
            'provenance': {
                "annotation-procedure": annotation_procedure,
                "annotation-role": "",
                "annotator-class": "UOregon",
                "annotator-id": "UOregon",
                "corpus-creation-date": "2020-03-19_10-37"
            }
        }
    )
    dev_wa_corpus_arb = EmptyCorpus(
        init_info={
            'lang': 'arabic',
            'corpus-id': 'word-alignment.LDC2014T05.LDC2014T10.LDC2014T14.LDC2014T19.LDC2014T22.arb.dev' + '.removed_incomplete' if
            opt[
                'remove_incomplete'] else '',
            'format-type': 'bp-corpus',
            'format-version': 'v9',
            'provenance': {
                "annotation-procedure": annotation_procedure,
                "annotation-role": "",
                "annotator-class": "UOregon",
                "annotator-id": "UOregon",
                "corpus-creation-date": "2020-03-19_10-37"
            }
        }
    )
    event_table, argument_table = load_tables()
    ####################################################
    opt['output_offsets'] = 1
    opt['hidden_eval'] = 1  # to output *_prob info
    wa_corpus_eng = Corpus(
        filepath='app/results_dir/english-wa-corpus.bp.json.sysfile',
        lang='english',
        parsing=False
    )
    ar_sentences = []
    for en_sentence in wa_corpus_eng.sentences:
        abstract_events = en_sentence.to_json_dict()['annotation-sets']['abstract-events']['events']
        wa_annots = en_sentence.wa_annotations

        entry_dict = copy.deepcopy(en_sentence.to_json_dict())
        entry_dict['annotation-sets'] = {}
        entry_dict['segment-text'] = wa_annots['ar-text']
        del entry_dict['wa-annotations']

        ar_sentence = Sentence(
            doc_id=en_sentence.doc_id,
            entry_dict=entry_dict,
            lang='arabic',
            parsing=False,
            normalize=False
        )
        num_events = 0
        for eid in abstract_events:
            abs_event = abstract_events[eid]
            if not is_good_event(abs_event, event_table, argument_table):
                continue

            anchor_word, agents, patients, anchor_word_en, agents_en, patients_en = get_arb_abstract_event(
                en_abstract_event=abs_event,
                wa_annotations=wa_annots)
            if anchor_word is None:
                continue
            else:
                anchor_word_ar = strip_punctuations(anchor_word)[0]
                if len(anchor_word) == 0:
                    continue

                agents_ar, patients_ar = [], []
                bad_a, bad_p = [], []
                for k in range(len(agents)):
                    norm_text = strip_punctuations(agents[k])[0]
                    if len(norm_text) > 0:
                        agents_ar.append(norm_text)
                    else:
                        bad_a.append(k)

                for k in range(len(patients)):
                    norm_text = strip_punctuations(patients[k])[0]
                    if len(norm_text) > 0:
                        patients_ar.append(norm_text)
                    else:
                        bad_p.append(k)

                agents_en = [agent for k, agent in enumerate(agents_en) if k not in bad_a]
                patients_en = [patient for k, patient in enumerate(patients_en) if k not in bad_p]

                assert len(agents_ar) == len(agents_en) and len(patients_ar) == len(patients_en)

            #####################################################################33
            event_id = f'event{len(ar_sentence.abstract_events) + 1}'
            anchor_ss_id = ar_sentence.add_span_set(span_strings=[anchor_word_ar])
            anchor_span_set = ar_sentence.span_sets[anchor_ss_id]
            agent_span_sets, patient_span_sets = [], []

            if len(agents_ar) > 0:
                agent_ss_id = ar_sentence.add_span_set(
                    span_strings=agents_ar)
            else:
                agent_ss_id = None
            if len(patients_ar) > 0:
                patient_ss_id = ar_sentence.add_span_set(
                    span_strings=patients_ar)
            else:
                patient_ss_id = None

            if agent_ss_id is not None:
                agent_span_sets.append(ar_sentence.span_sets[agent_ss_id])
            if patient_ss_id is not None:
                patient_span_sets.append(ar_sentence.span_sets[patient_ss_id])
            del abs_event['event_prob']
            del abs_event['argument_prob']
            abstract_event = AbstractEvent(
                event_id=event_id,
                helpful_harmful=abs_event['helpful-harmful'],
                material_verbal=abs_event['material-verbal'],
                anchor_span_set=anchor_span_set,
                agent_span_sets=agent_span_sets,
                patient_span_sets=patient_span_sets,
                en_ver={
                    'en-text': wa_annots['en-text'],
                    'event': abs_event,
                    'spansets': en_sentence.to_json_dict()['annotation-sets']['abstract-events']['span-sets']
                }
            )
            num_events += 1
            ar_sentence.add_abstract_event(abstract_event=abstract_event)
        if num_events > 0 and num_events == len(abstract_events):
            ar_sentences.append(ar_sentence)
        elif num_events > 0 and not opt['remove_incomplete']:
            ar_sentences.append(ar_sentence)
    print('English wa corpus: {} entries'.format(len(wa_corpus_eng.sentences)))
    opt['output_offsets'] = 0
    opt['hidden_eval'] = 0
    ########## split train/dev ###########
    train_size = int(0.7 * len(ar_sentences))
    train_ar_sentences = ar_sentences[: train_size]
    dev_ar_sentences = ar_sentences[train_size:]
    train_fpath = 'datasets/8d/update2/arabic-wa-corpus.train{}.bp.json'.format('.removed_incomplete' if opt[
        'remove_incomplete'] else '')
    dev_fpath = 'datasets/8d/update2/arabic-wa-corpus.dev{}.bp.json'.format('.removed_incomplete' if opt[
        'remove_incomplete'] else '')
    train_dev_fpath = 'datasets/8d/update2/arabic-wa-corpus{}.bp.json'.format('.removed_incomplete' if opt[
        'remove_incomplete'] else '')
    ######################################
    train_wa_corpus_arb.add_sentences(train_ar_sentences, normalize=False)
    print('Writing {} entries to file: {}'.format(len(train_ar_sentences), train_fpath))
    train_wa_corpus_arb.save(output_file=train_fpath)

    dev_wa_corpus_arb.add_sentences(dev_ar_sentences, normalize=False)
    print('Writing {} entries to file: {}'.format(len(dev_ar_sentences), dev_fpath))
    dev_wa_corpus_arb.save(output_file=dev_fpath)

    train_dev_wa_corpus_arb.add_sentences(train_ar_sentences + dev_ar_sentences, normalize=False)
    print('Writing {} entries to file: {}'.format(len(train_ar_sentences + dev_ar_sentences), train_dev_fpath))
    train_dev_wa_corpus_arb.save(output_file=train_dev_fpath)
    ############### show info ###############
    Corpus(
        filepath=train_fpath,
        lang='arabic',
        parsing=False
    )
    Corpus(
        filepath=dev_fpath,
        lang='arabic',
        parsing=False
    )
    Corpus(
        filepath=train_dev_fpath,
        lang='arabic',
        parsing=False
    )


def find_best_criterion():
    opt['output_offsets'] = 1
    opt['hidden_eval'] = 1  # to output *_prob info
    en_corpus = Corpus(
        filepath='app/results_dir/abstract-8d-inclusive.analysis.update2.bp.json.sysfile',
        lang='english',
        parsing=False
    )
    '''
    ARGUMENT_TAG_MAP = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2,
                  'O': 3, 'B-AGENT': 4, 'I-AGENT': 5,
                  'B-PATIENT': 6, 'I-PATIENT': 7,
                    # ********** conflict cases **********
                  'B-AGENT|B-PATIENT': 8, 'B-AGENT|I-PATIENT': 9, 'I-AGENT|B-PATIENT': 10, 'I-AGENT|I-PATIENT': 11
                    }
    '''
    argument_table = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
    '''
    EVENT_MAP = {
    '{}|{}'.format(UNKNOWN_EVENT_KEY, UNKNOWN_EVENT_KEY): 0,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, MATERIAL_KEY): 1,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, VERBAL_KEY): 2,
    '{}|{}'.format(UNKNOWN_EVENT_KEY, MATERIAL_VERBAL_KEY): 3,
    '{}|{}'.format(HARMFUL_KEY, UNKNOWN_EVENT_KEY): 4,
    '{}|{}'.format(HARMFUL_KEY, MATERIAL_KEY): 5,
    '{}|{}'.format(HARMFUL_KEY, VERBAL_KEY): 6,
    '{}|{}'.format(HARMFUL_KEY, MATERIAL_VERBAL_KEY): 7,
    '{}|{}'.format(HELPFUL_KEY, UNKNOWN_EVENT_KEY): 8,
    '{}|{}'.format(HELPFUL_KEY, MATERIAL_KEY): 9,
    '{}|{}'.format(HELPFUL_KEY, VERBAL_KEY): 10,
    '{}|{}'.format(HELPFUL_KEY, MATERIAL_VERBAL_KEY): 11,
    '{}|{}'.format(NEUTRAL_KEY, UNKNOWN_EVENT_KEY): 12,
    '{}|{}'.format(NEUTRAL_KEY, MATERIAL_KEY): 13,
    '{}|{}'.format(NEUTRAL_KEY, VERBAL_KEY): 14,
    '{}|{}'.format(NEUTRAL_KEY, MATERIAL_VERBAL_KEY): 15
    }
    '''
    event_table = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                   14: [], 15: []}
    score_map = parse_tsv_report(report_file='app/results_dir/abstract-8d-inclusive.analysis.update2.bp.json.tsv')

    score_sent_list = [(score_map.get(sent.entry_id, {
        'f1-event': 0,
        'f1-argument': 0,
        'f1-combined': 0
    })['f1-event'], score_map.get(sent.entry_id, {
        'f1-event': 0,
        'f1-argument': 0,
        'f1-combined': 0
    })['f1-argument'], sent) for sent in en_corpus.sentences]
    score_sent_list.sort(key=lambda x: -x[0])
    sents_event = [(f1_event, sent) for f1_event, _, sent in score_sent_list]
    score_sent_list.sort(key=lambda x: -x[1])
    sents_argument = [(f1_argument, sent) for _, f1_argument, sent in score_sent_list]

    ########### event ############
    for score, sent in sents_event:
        if score == 1.0:
            abstract_events = sent.to_json_dict()['annotation-sets']['abstract-events']['events']
            for eid in abstract_events:
                abs_event = abstract_events[eid]

                event_prob = abs_event['event_prob']  # event_prob.shape = [16, ]
                event_type_id = EVENT_MAP['{}|{}'.format(abs_event['helpful-harmful'], abs_event['material-verbal'])]
                event_table[event_type_id].append(event_prob[event_type_id])

    ########### argument #########
    for score, sent in sents_argument:
        if score == 1.0:
            abstract_events = sent.to_json_dict()['annotation-sets']['abstract-events']['events']
            for eid in abstract_events:
                abs_event = abstract_events[eid]
                token_probs = abs_event['argument_prob']  # argument_prob.shape = [num words, num BIO tags]

                for token_id in range(len(token_probs)):
                    arg_probs = token_probs[token_id]
                    arg_id = np.argmax(arg_probs)
                    argument_table[arg_id].append(arg_probs[arg_id])
    write_json(event_table, write_path='datasets/event_table.prob')
    write_json(argument_table, write_path='datasets/argument_table.prob')


if __name__ == '__main__':
    if opt['readers_mode'] == 1:
        create_observed_hidden_data(
            train_file='datasets/8d/update2/abstract-8d-inclusive.train.update2.bp.json'
        )
    elif opt['readers_mode'] == 2:
        create_wa_corpus_eng()
    elif opt['readers_mode'] == 3:
        create_wa_corpus_arb()
    elif opt['readers_mode'] == 4:
        find_best_criterion()
