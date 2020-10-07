# Copyright 2019 by Raytheon BBN Technologies Corp.
# All Rights Reserved.
import os, sys, logging
logger = logging.getLogger(__name__)
import argparse
import json, codecs
import os
import sys

from nlplingo.embeddings.word_embeddings import WordEmbeddingFactory
from nlplingo.nn.extractor import Extractor
from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.common.serialize_disk import ChunkWriter, WRITE_THRESHOLD
import pickle

def load_embeddings(params):
    """
    :return: dict[str : WordEmbeddingAbstract]
    """
    embeddings = dict()

    if 'embeddings' in params:
        embeddings_params = params['embeddings']
        word_embeddings = WordEmbeddingFactory.createWordEmbedding(
            embeddings_params.get('type', 'word_embeddings'),
            embeddings_params
        )
        embeddings['word_embeddings'] = word_embeddings
        print('Word embeddings loaded')

    if 'dependency_embeddings' in params:
        dep_embeddings_params = params['dependency_embeddings']
        dependency_embeddings = WordEmbeddingFactory.createWordEmbedding(
            dep_embeddings_params.get('type', 'dependency_embeddings'),
            dep_embeddings_params
        )
        embeddings['dependency_embeddings'] = dependency_embeddings
        print('Dependency embeddings loaded')

    return embeddings

def generate_candidates_and_featurize(docs, example_generator, feature_generator):
    candidates = example_generator.generate(docs)
    feature_generator.populate(candidates)
    return candidates

def generate_and_serialize_feature(example_generator, feature_generator, output_directory, train_docs, dev_docs, test_docs):
    train_examples = []
    dev_examples = []
    test_examples = []

    if train_docs:
        example_generator.train_dev_test_mode = "train"
        train_examples = generate_candidates_and_featurize(train_docs, example_generator, feature_generator)

    if dev_docs:
        example_generator.train_dev_test_mode = "dev"
        dev_examples = generate_candidates_and_featurize(dev_docs, example_generator, feature_generator)

    if test_docs:
        example_generator.train_dev_test_mode = "test"
        test_examples = generate_candidates_and_featurize(test_docs, example_generator, feature_generator)

    chunk_writer = ChunkWriter(WRITE_THRESHOLD, output_directory)
    if train_examples:
        chunk_writer.write_chunk_with_mode(train_examples, 'train')
    if dev_examples:
        chunk_writer.write_chunk_with_mode(dev_examples, 'dev')
    if test_examples:
        chunk_writer.write_chunk_with_mode(test_examples, 'test')
    chunk_writer.write_leftover()
    logging.info("Write times: %s", str(chunk_writer.write_times))

def split_docs(docs, mode_map):
    """
    Split the docs into train/dev/test docs.
    :param docs:
    :return:
    """
    train_docs = []
    dev_docs = []
    test_docs = []

    for doc in docs:
        assert(('train' in mode_map[doc.docid]) or ('dev' in mode_map[doc.docid]) or ('test' in mode_map[doc.docid]))
        if 'train' in mode_map[doc.docid]:
            train_docs.append(doc)
        if 'dev' in mode_map[doc.docid]:
            dev_docs.append(doc)
        if 'test' in mode_map[doc.docid]:
            test_docs.append(doc)

    return train_docs, dev_docs, test_docs

def main(args):
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    params['extractors'][0]["hyper-parameters"]["num_batches"] = int(args.num_batches)
    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)
    extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        try:
            extractor = Extractor(params, extractor_params, embeddings, False)
            extractors.append(extractor)
        except:
            raise RuntimeError('Extractor cannot be instantiated.')

    extractor = extractors[0]
    feature_generator = extractor.feature_generator
    example_generator = extractor.example_generator

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))

    docs = prepare_docs(args.filelist, embeddings, params)

    for doc in docs:
        doc.apply_domain(extractor.domain)

    with open(args.offset_dir + '/' + args.task + '.map', 'rb') as f:
        mode_map = pickle.load(f)

    train_docs, dev_docs, test_docs = split_docs(docs, mode_map)

    generate_and_serialize_feature(
        example_generator,
        feature_generator,
        args.output_directory,
        train_docs,
        dev_docs,
        test_docs
    )

def parse(args_list):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('params')
    arg_parser.add_argument('task')
    arg_parser.add_argument('filelist')
    arg_parser.add_argument('output_directory')
    arg_parser.add_argument('offset_dir')
    arg_parser.add_argument('num_batches')
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