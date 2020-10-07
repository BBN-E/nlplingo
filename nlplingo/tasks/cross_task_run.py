import logging
logger = logging.getLogger(__name__)
from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.tasks.eventrelation.mention_pool_filter import mention_pool_filter_from_memory
from nlplingo.common.serialize_disk import load_from_serialized
import os

def generate_binary_triplets_with_relations_from_candidates(candidates):
    triplets = []
    for candidate in candidates:
        triplets.append(candidate.to_triplet_with_relation())
    return triplets

def train_bert_mention(params, word_embeddings, extractor, serialize_list, in_memory_data_feature_generator, from_serialized_feature_loader, k_partitions=None, partition_id=None):
    feature_generator = extractor.feature_generator
    example_generator = extractor.example_generator
    extraction_model = extractor.extraction_model

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(extraction_model)={}'.format(type(extraction_model)))

    if serialize_list is None:
        train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
        test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)

        for doc in train_docs:
            doc.apply_domain(extractor.domain)
        for doc in test_docs:
            doc.apply_domain(extractor.domain)

        (train_examples, train_data, train_data_list, train_label) = in_memory_data_feature_generator(
            example_generator,
            train_docs,
            feature_generator
        )
        print(train_label)

        (dev_examples, dev_data, dev_data_list, dev_label) = in_memory_data_feature_generator(
            example_generator,
            test_docs,
            feature_generator
        )
    else:
        train_candidates, dev_candidates, test_candidates = load_from_serialized(serialize_list, k_partitions, partition_id)
        (train_examples, train_data, train_data_list, train_label) = from_serialized_feature_loader(example_generator, feature_generator, train_candidates)
        (dev_examples, dev_data, dev_data_list, dev_label) = from_serialized_feature_loader(example_generator, feature_generator, dev_candidates)
        # (test_examples, test_data, test_data_list, test_label) = from_serialized_feature_loader(example_generator, feature_generator, test_candidates)
        # test_tuple = (test_examples, test_data, test_data_list, test_label)

    if extraction_model.hyper_params.encoder == 'bert_mention':
        train_triplets = generate_binary_triplets_with_relations_from_candidates(train_examples)
        dev_triplets = generate_binary_triplets_with_relations_from_candidates(dev_examples)
        interm_dir = extraction_model.hyper_params.save_model_path + '/interm'
        if not os.path.isdir(interm_dir):
            os.makedirs(interm_dir)
        train_path = interm_dir + '/train_mention_pool.json'
        dev_path = interm_dir + '/dev_mention_pool.json'
        mention_pool_filter_from_memory(train_triplets, train_path)
        mention_pool_filter_from_memory(dev_triplets, dev_path)
        extraction_model.fit_txt(train_path, dev_path, None)
    else:
        raise Exception('No other models supported currently.')