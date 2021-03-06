# Copyright 2020 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import argparse
import codecs
import os
import logging
import pickle
import sys

from knowledge_base.internal_ontology import Ontology


logger = logging.getLogger(__name__)

EVENT_TEMPLATE = '<Event type="{}">\n</Event>'
EVENT_NA = "NA"


def get_relations_from_yaml(yaml_file):
    final_set = set()
    ontology = Ontology()
    ontology.load_from_internal_yaml(yaml_file)
    nodes = [ontology.get_root()]
    while nodes:
        node = nodes.pop()
        final_set.add(node.get_name())
        nodes.extend(node.get_children())
    return final_set


def main(args):
    """
    Add all relations into an ontology.
    :param args:
    :return:
    """
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    final_set = get_relations_from_yaml(args.yaml_file)
    #final_set.add(EVENT_NA)

    with open(args.output_directory + '/ontology.txt', 'w') as fp:
        for relation in sorted(final_set):
            fp.write(EVENT_TEMPLATE.format(relation) + "\n")


def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--yaml_file', required=True)
    arg_parser.add_argument('--output_directory', required=True)
    _args = arg_parser.parse_args(args_list)
    return _args


if __name__ == '__main__':
    log_format = ('[%(asctime)s] {P%(process)d:%(module)s:%(lineno)d} '
                  '%(levelname)s - %(message)s')
    try:
        logging.basicConfig(
            level=logging.getLevelName(
                os.environ.get('LOGLEVEL', 'INFO').upper()),
            format=log_format)
    except ValueError as e:
        logging.error(
            "Unparseable level {}, will use default {}.".format(
                os.environ.get('LOGLEVEL', 'INFO').upper(),
                logging.root.level))
        logging.basicConfig(format=log_format)
    main(parse(sys.argv[1:]))
