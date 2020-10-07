# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import json, codecs
import os
import sys
import pickle
import re

from nlplingo.common import io_utils
from nlplingo.annotation.ingestion import parse_filelist_line
from xml.dom import minidom

def read_lines(filename):
    file_lines = open(filename)
    lines = []
    for line in file_lines:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        lines.append(line)
    return lines

def update_doc_to_mode_map(file_list, doc_to_mode_map, mode):
    """
    Read the file_list in, parsing the SerifXML docid and setting the docid to the proper mode in the map.
    :param file_list: 
    :param doc_to_mode_map: 
    :param mode: 
    :return: 
    """
    number_of_lines = 4
    for file_index, line in enumerate(file_list):
        input_file_type = parse_filelist_line(line)
        a_file = open(input_file_type.serif_file)
        docid = None
        for i in range(number_of_lines):
            line = a_file.readline()
            if 'Document' in line and 'docid' in line:
                elems = line.split()
                for elem in elems:
                    if 'docid' in elem:
                        match_lst = re.findall(r'docid\="(.+)"', elem)
                        if len(match_lst) > 1 or len(match_lst) < 1:
                            raise Exception('docid not valid')
                        docid = match_lst[0]
                        break
            if docid is not None:
                break

        if docid is None:
            raise Exception('docid is not available in SerifXML.')
        # print('docid', docid)
        # xmldoc = minidom.parse()
        # itemlist = xmldoc.getElementsByTagName('Document')
        # docid = itemlist[0].attributes['docid'].value

        if docid not in doc_to_mode_map:
            doc_to_mode_map[docid] = set()

        doc_to_mode_map[docid].add(mode)

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    data = json.load(codecs.open(args.params, 'r', 'utf-8'))
    # create a docid to mode (train/dev/test) map
    doc_to_mode_map = dict()

    if 'train' in data['data']:
        train_list = data['data']['train']['filelist']
        train_filelist = io_utils.read_file_to_list(train_list)
        update_doc_to_mode_map(train_filelist, doc_to_mode_map, 'train')

    if 'dev' in data['data']:
        dev_list = data['data']['dev']['filelist']
        dev_filelist = io_utils.read_file_to_list(dev_list)
        update_doc_to_mode_map(dev_filelist, doc_to_mode_map, 'dev')

    if 'test' in data['data']:
        test_list = data['data']['test']['filelist']
        test_filelist = io_utils.read_file_to_list(test_list)
        update_doc_to_mode_map(test_filelist, doc_to_mode_map, 'test')

    # write map
    with open(args.output_directory + '/' + args.task + '.map', 'wb') as f:
        pickle.dump(doc_to_mode_map, f)

    # write the concatatenated train/dev/test lines
    train_lines = []
    dev_lines = []
    test_lines = []

    if 'train' in data['data']:
        train_lines = read_lines(train_list)

    if 'dev' in data['data']:
        dev_lines = read_lines(dev_list)

    if 'test' in data['data']:
        test_lines = read_lines(test_list)

    final_lines = train_lines + dev_lines + test_lines

    with open(args.output_directory + '/' + args.task + '.list', 'w') as fp:
        for line in final_lines:
            fp.write(line + "\n")

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('params')
    arg_parser.add_argument('task')
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
