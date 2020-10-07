# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import json, codecs
import os
import sys
import random

def process_json(json_file, final_json_objs):
    """
    Add json_objs from json_file to the end of final_json_objs.
    :param json_file:
    :param final_json_objs:
    :return:
    """
    final_json_objs.extend(json.load(codecs.open(json_file, 'r', 'utf-8')))

HUMAN_LABEL = 'HumanLabeled'
def process_labels(label_json_objs, learnit_pattern_bound):
    """
    Perform sampling by LearnIt patterns.
    :param label_json_objs:
    :param learnit_pattern_bound:
    :return:
    """
    relation_count = dict()
    for json_obj in label_json_objs:
        if json_obj['relation'] not in relation_count:
            relation_count[json_obj['relation']] = 0
        relation_count[json_obj['relation']] += 1
    print(relation_count)

    """
    pattern_count = dict()
    for json_obj in label_json_objs:
        if json_obj['pattern'] not in pattern_count:
            pattern_count[json_obj['pattern']] = 0
        pattern_count[json_obj['pattern']] += 1
    print(pattern_count)
    """

    relation_pattern_instance_dict = {}
    human_json_objs = set()

    for i in range(len(label_json_objs)):
        json_obj = label_json_objs[i]
        relation = json_obj['relation']

        if relation not in relation_pattern_instance_dict:
            relation_pattern_instance_dict[relation] = {}

        pattern = json_obj['pattern']
        if HUMAN_LABEL in pattern:
            human_json_objs.add(i)
            continue

        if pattern not in relation_pattern_instance_dict[relation]:
            relation_pattern_instance_dict[relation][pattern] = set()
        relation_pattern_instance_dict[relation][pattern].add(i)

    ct = 0
    num_patterns = 0
    for relation in relation_pattern_instance_dict:
        for pattern in relation_pattern_instance_dict[relation]:
            # print('pattern', pattern)
            num_patterns += 1
            ct += len(relation_pattern_instance_dict[relation][pattern])
    print('ct', ct)
    print('num_patterns', num_patterns)
    assert(ct == len(label_json_objs) - len(human_json_objs))

    examples_per_learnit_pattern = int(learnit_pattern_bound)
    print('learnit_pattern_bound', examples_per_learnit_pattern)
    automatic_learnit_instance_set = set()
    for relation in relation_pattern_instance_dict:
        for pattern in relation_pattern_instance_dict[relation]:
            num_examples = min(len(relation_pattern_instance_dict[relation][pattern]),
                               examples_per_learnit_pattern)
            automatic_learnit_instance_set.update(
                random.sample(relation_pattern_instance_dict[relation][pattern], num_examples))

    # Actually produce final JSON file
    complete_list = []
    non_na_count = 0
    na_count = 0
    print('human samples', len(human_json_objs))
    print('learnit samples', len(automatic_learnit_instance_set))
    human_na_count = 0
    for i in range(len(label_json_objs)):
        json_obj = label_json_objs[i]
        if (i in automatic_learnit_instance_set) or (i in human_json_objs):
            if (i in human_json_objs):
                if json_obj['relation'] == 'NA':
                    human_na_count += 1
            # print(json_obj['pattern'])
            del json_obj['pattern']
            if json_obj['relation'] == 'NA':
                na_count += 1
            else:
                non_na_count += 1
            # print(json_obj)
            complete_list.append(json_obj)
    print('human_na_count', human_na_count)
    print('non_na_count', non_na_count)
    print('na_count', na_count)
    return complete_list, non_na_count, na_count

def create_relation_dict(json_objs):
    """
    Perform deduplication and split into relation categories.
    :param json_objs:
    :return:
    """
    unique_set = set()
    dedup_json_obj_dict = dict()
    for d in json_objs:
        str_rep = str(dict(sorted(d.items())))
        if str_rep not in unique_set:
            unique_set.add(str_rep)
            relation = d['relation']
            if relation not in dedup_json_obj_dict:
                dedup_json_obj_dict[relation] = []
            dedup_json_obj_dict[relation].append(d)

    for relation in dedup_json_obj_dict:
        random.shuffle(dedup_json_obj_dict[relation])
    return dedup_json_obj_dict

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    """
    Process LearnIt labels (human labeled + LearnIt pattern extractors).
    """
    i = open(args.label_file)
    label_json_objs = []
    for line in i:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        process_json(line, label_json_objs)

    # print(json_obj['relation'])
    print('num_ori_label_json_objs', len(label_json_objs))
    sampled_label_objs, non_na_count, na_count = process_labels(label_json_objs, args.learnit_pattern_bound)
    print('sampled_label_objs', len(sampled_label_objs))

    """
    Add NA to non-NA samples.
    """
    na_to_non_na_ratio = int(args.na_to_non_na_ratio)
    print('na_to_non_na_ratio', na_to_non_na_ratio)
    num_na_to_add = (non_na_count * na_to_non_na_ratio) - na_count
    print('num_na_to_add', num_na_to_add)
    num_models_to_train = int(args.num_models_to_train)
    final_json_obj_dict = dict()

    if (num_na_to_add > 0):
        # process na file
        i = open(args.na_file)
        na_json_objs = []
        for line in i:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            process_json(line, na_json_objs)
        print('num_ori_na_json_objs', len(na_json_objs))
        # process label file
        if (num_na_to_add > len(na_json_objs)):
            raise Exception('Not enough NA samples for given na_to_non_na_ratio.')
    else:
        print("The na_to_non_na_ratio is already satisfied in the input LearnIt SerifXML set. No additional NA examples will be added.")

    for j in range(num_models_to_train):
        final_json_obj_dict[j] = list()
        if (num_na_to_add > 0):
            random.shuffle(na_json_objs)
            final_json_obj_dict[j].extend(na_json_objs[:num_na_to_add])
        final_json_obj_dict[j].extend(sampled_label_objs)
        relation_dict = create_relation_dict(final_json_obj_dict[j])
        train_ratio = float(args.train_ratio)
        print('train_ratio', train_ratio)
        val_ratio = (1 - train_ratio) / (2.0)

        train_examples = []
        val_examples = []
        test_examples = []
        for relation in relation_dict:
            num_relations = len(relation_dict[relation])
            num_train = int(num_relations*train_ratio)
            num_val = int(num_relations*val_ratio)
            num_test = num_relations - (num_train + num_val)
            print('relation', relation)
            print('num_train', num_train)
            print('num_val', num_val)
            print('num_test', num_test)
            train_examples.extend(relation_dict[relation][:num_train])
            val_examples.extend(relation_dict[relation][num_train:num_train+num_val])
            test_examples.extend(relation_dict[relation][num_train+num_val:])

        random.shuffle(train_examples)
        random.shuffle(val_examples)
        random.shuffle(test_examples)

        with open(args.output_directory + '/train_' + str(j) + '.json', 'w') as fp:
            for example in train_examples:
                fp.write(str(example) + "\n")

        with open(args.output_directory + '/val_'  + str(j) + '.json', 'w') as fp:
            for example in val_examples:
                fp.write(str(example) + "\n")

        with open(args.output_directory + '/test_'  + str(j) + '.json', 'w') as fp:
            for example in test_examples:
                fp.write(str(example) + "\n")

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('na_file')
    arg_parser.add_argument('label_file')
    arg_parser.add_argument('output_directory')
    arg_parser.add_argument('learnit_pattern_bound')
    arg_parser.add_argument('na_to_non_na_ratio')
    arg_parser.add_argument('train_ratio')
    arg_parser.add_argument('num_models_to_train')
    _args = arg_parser.parse_args(args_list)
    return _args

if __name__ == '__main__':
    log_format = '[%(asctime)s] {P%(process)d:%(module)s:%(lineno)d} %(levelname)s - %(message)s'
    try:
        logging.basicConfig(level=logging.getLevelName(os.environ.get('LOGLEVEL', 'INFO').upper()),
                            format=log_format)
    except ValueError as e:
        logging.error(
            "Unparseable level {}, will use default {}.".format(os.environ.get('LOGLEVEL', 'INFO').upper(),
                                                                logging.root.level))
        logging.basicConfig(format=log_format)
    main(parse(sys.argv[1:]))