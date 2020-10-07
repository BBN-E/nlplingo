# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import json, codecs
import os
import sys

import time

def convert_to_nlplingo(json_file, output_dir):
    json_objs_converted = []
    json_objs = json.load(codecs.open(json_file, 'r', 'utf-8'))
    base = os.path.basename(json_file)
    base_file = os.path.splitext(base)[0]
    for json_obj in json_objs:
        json_obj_new = {'text': json_obj['sentence'].encode('utf-8').decode('utf-8'),
                        'h': {'pos': (json_obj['arg1_anchor_span_list'][0][0] - json_obj['sentStartCharOff'],
                                      json_obj['arg1_anchor_span_list'][0][1] - json_obj['sentStartCharOff'] + 1)},
                        't': {'pos': (json_obj['arg2_anchor_span_list'][0][0] - json_obj['sentStartCharOff'],
                                      json_obj['arg2_anchor_span_list'][0][1] - json_obj['sentStartCharOff'] + 1)},
                        'relation': json_obj['relation']}
        json_objs_converted.append(json_obj_new)

    json.dump(json_objs_converted,open(output_dir + '/' + base_file + '_converted.json','w'),indent=4, separators=(',', ': '))

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    files_to_process = []
    # Assume batch file
    i = open(args.input_file)
    for line in i:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        files_to_process.append(line)

    for idx, input_file in enumerate(files_to_process):
        logger.info("({}/{})Loading: {}".format(idx + 1, len(files_to_process), input_file))
        convert_to_nlplingo(input_file, args.output_directory)

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file')
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