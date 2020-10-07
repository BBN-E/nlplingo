# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import os
import sys
import pickle
import json, codecs

from knowledge_base.internal_ontology import Ontology


EER_TEMPLATE = '<EntityRelation type="{}">'
EER_NA = "NA"


def get_relations_from_jsons(json_file):
    i = open(json_file)
    final_set = set()
    for line in i:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue

        with open(line, 'rb') as handle:
            json_objs = json.load(codecs.open(line, 'r', 'utf-8'))
            for json_obj in json_objs:
                final_set.add(json_obj['relation'])
    i.close()
    return final_set


def get_relations_from_yaml(yaml_file):
    final_set = set()
    ontology = Ontology()
    ontology.load_from_internal_yaml(yaml_file)
    nodes = [ontology.get_root()]
    while nodes:
        node = nodes.pop()
        name = node.get_name()
        if name == 'Binary-Entity' or name == 'Binary-Event':
            pass
        else:
            final_set.add(name)
        nodes.extend(node.get_children())
    return final_set


def main(args):
    """
    Add all relations into an ontology.
    Two modes are supported:
    1. If a json file is supplied,
    :param args:
    :return:
    """
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    if args.json_file is not None:
        final_set = get_relations_from_jsons(args.json_file)
    elif args.yaml_file is not None:
        final_set = get_relations_from_yaml(args.yaml_file)

    final_set.add(EER_NA) # TODO: add in original ontology file? currently NA is added manually

    with open(args.output_directory + '/ontology.txt', 'w') as fp:
        for relation in sorted(final_set):
            fp.write(EER_TEMPLATE.format(relation) + "\n")

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--json_file')
    arg_parser.add_argument('--yaml_file')
    arg_parser.add_argument('--output_directory', required=True)
    _args = arg_parser.parse_args(args_list)
    has_input = (_args.json_file is not None or _args.yaml_file is not None)
    both_inputs = (_args.json_file is not None and _args.yaml_file is not None)
    assert has_input and not both_inputs
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
