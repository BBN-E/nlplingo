# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import json, codecs
import os
import sys
import glob
import re

def process_score_file(score_file, score_file_dict):
    i = open(score_file)

    line_lst = []
    for line in i:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        line_lst.append(line)

    score_file_dict[line_lst[1]] = float(line_lst[0])

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    i = open(args.test_score_list)
    score_file_dict = dict()
    for line in i:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        process_score_file(line, score_file_dict)

    final_model_path = max(score_file_dict, key=score_file_dict.get)

    print('final_model_path', final_model_path)
    json_objs = json.load(codecs.open(args.initial_json, 'r', 'utf-8'))
    json_objs['extractors'][0]['hyper-parameters']['opennre_ckpt'] = final_model_path
    json.dump(json_objs,open(args.output_directory + '/decoding_config.json','w'),indent=4, separators=(',', ': '))

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('initial_json')
    arg_parser.add_argument('test_score_list')
    arg_parser.add_argument('output_directory')
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