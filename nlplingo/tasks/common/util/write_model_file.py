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

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    json_config = json.load(codecs.open(args.params, 'r', 'utf-8'))
    model_file_path = json_config['extractors'][0]['model_file']

    with open(args.output_directory + '/' + args.task + '.model_file', 'w') as f:
        f.write(model_file_path)

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--params', required=True)
    arg_parser.add_argument('--task', required=True)
    arg_parser.add_argument('--output_directory', required=True)
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