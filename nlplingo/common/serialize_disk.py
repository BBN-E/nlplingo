from __future__ import absolute_import
from __future__ import division
import gc
import sys
import os
import pickle
import time
import logging

import numpy as np
logger = logging.getLogger(__name__)

from nlplingo.common import io_utils

WRITE_THRESHOLD = 10 * 1e9 # default write threshold is 10 GB

# When this is set to 2, `divide_chunks` generates multiple lists , each of len 2.
# For each chunk of 10GB examples (say [file1.npz, file2.npz, file3.npz]),
# NUM_BIG_CHUNKS defines how many 10GB chunks are loaded in memory.
# In this case, since NUM_BIG_CHUNKS is 2, 2 files will be loaded and hence 20GB will be loaded in memory).
NUM_BIG_CHUNKS = 2

class ChunkWriter(object):
    """
    ChunkWriter writes a list of chunks of data to disk.
    A list of chunks is written to disk_path whenever write_threshold is exceeded.
    A counter is internally incremented for the filenames.
    """
    def __init__(self, write_threshold, disk_path):
        self.write_threshold = write_threshold
        self.curr_chunk_size = 0
        self.write_times = []
        self.chunks = []
        self.count = 0
        self.disk_path = disk_path

    def write(self, chunk):
        """
        Write a chunk to disk.
        :param chunk: an arbitrary Python object
        """

        # E.g. chunk might be: list[nlplingo.tasks.eventargument.example.EventArgumentExample]
        chunk_size = get_obj_size(chunk)
        self.chunks.extend(chunk)
        self.curr_chunk_size += chunk_size

        if self.curr_chunk_size > self.write_threshold:
            start_time = time.time()
            write_to_disk(self.chunks, self.disk_path, self.count)
            self.write_times.append(time.time() - start_time)
            self.count += 1
            self.chunks = []
            self.curr_chunk_size = 0

    def write_chunk_with_mode(self, chunk, train_dev_test_mode):
        """
        Write a chunk to disk.
        :param chunk: an arbitrary Python object
        """

        # E.g. chunk might be: list[nlplingo.tasks.eventargument.example.EventArgumentExample]
        for example in chunk:
            example.train_dev_test_mode = train_dev_test_mode

        chunk_size = get_obj_size(chunk)
        self.chunks.extend(chunk)
        self.curr_chunk_size += chunk_size

        if self.curr_chunk_size > self.write_threshold:
            start_time = time.time()
            write_to_disk(self.chunks, self.disk_path, self.count)
            self.write_times.append(time.time() - start_time)
            self.count += 1
            self.chunks = []
            self.curr_chunk_size = 0

    def write_leftover(self):
        """
        Write whatever is left in chunks (since it is possible chunks does not exceed write_threshold in the end).
        :return:
        """
        if self.chunks:
            start_time = time.time()
            write_to_disk(self.chunks, self.disk_path, self.count)
            self.write_times.append(time.time() - start_time)

def write_to_disk(example_lst, disk_path, count):
    """
    Write an NLPLingo example to disk_path, with filename count.npz.
    :param example_lst: list of nlplingo examples
    :param disk_path: str
    :param count: int
    :return:
    """
    os.makedirs(disk_path, exist_ok=True)

    len_file = disk_path + '/' + str(count) + '.len'
    npz_file = disk_path + '/' + str(count) + '.npz'
    lbl_file = disk_path + '/' + str(count) + '.lbl'

    # If we are dealing with event arguments, then `len(example_lst)` is the number of EventArgumentExample instances.
    # It is necessary to save this information in order to properly set the Keras fit_generator function,
    # which requires knowing the number of steps to run an entire epoch.
    # Since we no longer save all the examples at once into memory, we need to save this information to disk somewhere
    # and reconstruct the total number of training/validation examples
    with open(len_file, 'wb') as f:
        pickle.dump(len(example_lst), f)

    with open(npz_file, 'wb') as f:
        pickle.dump(example_lst, f)

    labels = [example.label for example in example_lst]
    labels = np.asarray(labels)

    with open(lbl_file, 'wb') as f:
        pickle.dump(labels, f)

def ensure_dir(d, verbose=True):
    """
    Creates directory d if it does not exist.
    :param d: directory
    :param verbose: print
    :return:
    """
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def divide_chunks(l, n):
    """
    :param l: a list
    :param n: chunk size
    :return: a generator splitting l into sublists of size n
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_big_chunk(big_chunk_file_list):
    """
    :param big_chunk_file_list: a list of filenames, composing the big chunk
    :return: loaded big chunk, which is a list of chunks (one per filename)
    """
    chunks = []
    for file_name in big_chunk_file_list:
        with open(file_name, 'rb') as f:
            npz_data = pickle.load(f)
        chunks.append(npz_data)
    return chunks

def load_from_serialized(file_list, k_partitions=None, partition_id=None):
    """
    :param file_list: a list of filenames, composed of serialized npz files
    :param k_partitions: if running k-folds validation, the value of k.
    :param partition_id: which partition is used for testing in k-folds validation.
    :return: all examples loaded from the entire list of serialized npz files, split into three categories
    """

    train_lst = []
    dev_lst = []
    test_lst = []

    file_list_parent = os.path.dirname(file_list)
    file_list = io_utils.read_file_to_list(file_list)
    chunk_lst = load_big_chunk(file_list)
    for chunk in chunk_lst:
        for example in chunk:
            if example.train_dev_test_mode == 'train':
                train_lst.append(example)
            elif example.train_dev_test_mode == 'dev':
                dev_lst.append(example)
            elif example.train_dev_test_mode == 'test':
                test_lst.append(example)
            else:
                raise Exception('Fatal error: nlplingo Datapoint was not serialized with an appropriate train_dev_test mode')

    # k-folds partitioning: deal each example in the train and dev sets into one
    # of k folds deterministically.  One fold is used for testing and the rest
    # for both training and validation.
    if k_partitions is not None and partition_id is not None:
        assert k_partitions > 2
        folds = multilabel_stratified_sample(k_partitions,
                                             train_lst + dev_lst + test_lst)

        # assign folds to train+dev or test lists
        train_lst = []
        dev_lst = []
        test_lst = []
        for fold_id, fold in enumerate(folds):
            if fold_id == partition_id:
                test_lst = fold
                logger.info("Fold {} is the specified test partition".format(fold_id))
            elif fold_id == (partition_id + 1) % k_partitions:
                dev_lst = fold
                logger.info("Fold {} is the specified validation partition".format(fold_id))
            else:
                train_lst.extend(fold)
                logger.info("Fold {} is in the specified training partition".format(fold_id))

            # Re-serialize partitioned data for review / simpler reuse
            partitioned_directory = "{}/partitioned".format(file_list_parent)
            write_to_disk(fold, partitioned_directory, fold_id)

    return train_lst, dev_lst, test_lst


def multilabel_stratified_sample(k_partitions, tune_lst):
    """
    This function assigns each candidate to one fold, stratifying by type
    according to datapoint.label.

    It uses a modified version of Algorithm 1 from "On the Stratification of
    Multi-Label Data," Sechidis, Tsoumakas, & Vlahavas 2011
    :param k_partitions: an int, the number of partitions to create.
    :param tune_lst: a list of Datapoints
    :return: a list of k_partitions lists of Datapoints
    """
    folds = [list() for k in range(k_partitions)]

    def label_from_1darray(datapoint):
        return datapoint.label

    def label_from_str_fn_builder(label_map):
        """creates a 1darray with a consistent mapping from labels to indices"""
        def label_from_str_or_int(datapoint):
            label_array = np.zeros(len(label_map))
            label_array[label_map[datapoint.label]] = 1
            return label_array
        return label_from_str_or_int

    stratify = True
    label_type = type(tune_lst[0].label)
    if label_type is str or label_type is int:
        distinct_labels = dict()
        for candidate in tune_lst:
            label = candidate.label
            if label not in distinct_labels:
                distinct_labels[label] = len(distinct_labels)
        # returns 1d array of ints
        label_array = label_from_str_fn_builder(distinct_labels)

    elif label_type is np.ndarray:
        shape = tune_lst[0].label.shape
        if len(shape) == 1:
            num_labels = len(tune_lst[0].label)
            # returns 1d array of ints
            label_array = label_from_1darray

        else:
            logger.info("Datapoint.label ndarray shape {} is not acceptable for stratified partitioning".format(shape))
            stratify = False
    else:
        logger.info("Datapoint.label type {} is not acceptable for stratified partitioning".format(label_type))
        stratify = False

    if not stratify:
        logger.info("Partitioning without stratification...")
        for i, candidate in enumerate(tune_lst):
            fold = i % k_partitions
            folds[fold].append(candidate)
        logger.info("FOLDS: {}".format("; ".join(
            "{}: {}".format(i, len(f)) for i, f in enumerate(folds))))
        return folds

    np.random.seed(label_array(tune_lst[0]))  # call to `seed` must remain

    # Overlapping subsets of candidates having each label:
    # For deterministic access later on (when looping over all candidates
    # having a particular label), we will sort by an easily-ordered identifier.
    # The Datapoint is the key, as we want O(1) lookup of candidates-per-label
    # instead of O(N) when we pop each candidate we have assigned from each
    # label subset at the end.
    label_subsets = [dict() for i in range(num_labels)]
    uid = 0
    for candidate in tune_lst:
        for positive_label in np.argwhere(label_array(candidate) == 1).flatten():
            label_subsets[positive_label][candidate] = uid  # D_i
            uid += 1

    num_unassigned_candidates_per_label = np.asarray([len(label_subset) for label_subset in label_subsets])  # |D_i|
    num_assigned_candidates_per_fold = [0 for k in range(k_partitions)]  # c_j
    num_assigned_candidates_per_label_per_fold = [[0 for l in range(num_labels)] for k in range(k_partitions)]

    while np.sum(num_unassigned_candidates_per_label) > 0:
        # "Find the label with the fewest (but at least one) remaining
        # examples, breaking ties randomly".
        # (we need this to be deterministic, so we seed using the first label.)
        nonzero_unassigned_candidates_per_label = np.ma.MaskedArray(num_unassigned_candidates_per_label, num_unassigned_candidates_per_label < 1)
        label = np.ma.argmin(nonzero_unassigned_candidates_per_label)
        candidates_with_smallest_label = label_subsets[label]

        logger.info("Assigning {} candidates with label {} (candidates remaining: {})"
                    .format(len(candidates_with_smallest_label), label, np.sum(num_unassigned_candidates_per_label )))

        for candidate, uid in sorted(candidates_with_smallest_label.items(),
                                     key=lambda x: x[1]):
            # "Find the subset(s) with the largest number of desired
            # examples for this label, breaking ties by considering the
            # largest number of desired examples, breaking further ties
            # randomly".

            # use fold with fewest assigned candidates for this label
            num_assigned_candidates_with_label = [k[label] for k in num_assigned_candidates_per_label_per_fold]
            folds_with_fewest_of_label = np.argwhere(
                num_assigned_candidates_with_label
                == np.amin(num_assigned_candidates_with_label)).flatten()
            if len(folds_with_fewest_of_label) == 1:
                target_fold = folds_with_fewest_of_label[0]

            # use fold with fewest candidates overall from among the tied folds
            else:
                num_assigned_candidates_with_label = [num_assigned_candidates_per_fold[k] for k in folds_with_fewest_of_label]
                smallest_fold_idxs = np.argwhere(
                    num_assigned_candidates_with_label
                    == np.amin(num_assigned_candidates_with_label)).flatten()
                tied_folds_with_fewest_candidates_overall = (
                    folds_with_fewest_of_label[smallest_fold_idxs])

                if len(tied_folds_with_fewest_candidates_overall) == 1:
                    target_fold = tied_folds_with_fewest_candidates_overall[0]

                # select randomly from folds that tied twice
                else:
                    target_fold = np.random.choice(tied_folds_with_fewest_candidates_overall)

            # 1. assign this candidate to the appropriate fold
            folds[target_fold].append(candidate)

            # 2. remove this candidate from all subsets that contain it
            # 3. update example counters used to select folds c_{i,j} & c_j
            for positive_label in np.argwhere(label_array(candidate) == 1).flatten():
                label_subsets[positive_label].pop(candidate)
                num_unassigned_candidates_per_label[positive_label] -= 1
                num_assigned_candidates_per_label_per_fold[target_fold][positive_label] += 1
            num_assigned_candidates_per_fold[target_fold] += 1

    logger.info("FOLDS: {}".format("; ".join(
        "{}: {}".format(i, len(f)) for i, f in enumerate(folds))))

    return folds

def load_lens(len_file_list):
    """
    Loads the number of examples in len_file_list. This is needed for keras fit_generator
    :param len_file_list: a list of length filenames, recording number of examples in each chunk
    :return: int, total length across all files
    """
    total_len = 0
    for file_name in len_file_list:
        with open(file_name, 'rb') as f:
            len_data = pickle.load(f)
        total_len += len_data
    return total_len

def load_class_weights(lbl_file_list):
    """
    Computes class weights, across all training example labels.
    :param lbl_file_list:  list of training label filenames
    :return: class weights
    """
    # CLASS WEIGHT CODE
    labels = None
    for file_name in lbl_file_list:
        with open(file_name, 'rb') as f:
            if labels is None:
                labels = pickle.load(f)
            else:
                labels = np.concatenate((labels, pickle.load(f)), axis=0)

    count_per_label = np.sum(labels, axis=0)
    # count_all_labels = float(np.sum(count_per_label))
    # total_over_count = count_all_labels / count_per_label

    # CODE FOR INVERSE LOG FREQUENCY CLASS WEIGHTS
    inverse_log_freqs = 1 / np.log(count_per_label)
    class_weights = {i: weight for i, weight in enumerate(inverse_log_freqs)}

    logger.debug("CLASS WEIGHTS:")
    logger.debug("{}".format(sorted(class_weights.items(), key=lambda x: x[0])))
    logger.debug("CLASS FREQUENCIES:")
    logger.debug(" ".join(["{}: {}".format(i, x) for i, x in enumerate(count_per_label)]))
    return class_weights

def get_obj_size(obj):
    """
    Recurse into `obj` and count all pieces of memory inside `obj` non-redundantly.
    Below, `marked` stores object references that have already been counted.
    Adapted from: https://stackoverflow.com/questions/13530762/how-to-know-bytes-size-of-python-object-like-arrays-and-dictionaries-the-simp
    :param obj: an arbitrary Python object
    :return: size of obj in bytes
    """

    # `id` is a Python function that records the memory address of the Python object
    #
    # `marked` records those objects that have already been counted and thus prevents counting those pieces of memory
    # that have already been counted previously in this function.
    # For example, if you have a list containing [obj1, obj1], after counting the first obj1,
    # it is not necessary to count the underlying memory of obj1 again, since pointers are being used.
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        # Count memory needed for top level objects.
        sz += sum(map(sys.getsizeof, obj_q))    # `map` applies sys.getsizeof to each obj_q

        # Look up all objects referred to by the object in obj_q (one level below the top layer).
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter out objects that have already been marked.
        # dict notation will prevent repeated objects from being included.
        # Avoid including types in the memory count with `not isinstance(o,type)`.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be ones that have not been marked
        obj_q = new_refr.values()

        # Update `marked` with ids so we will not traverse them again
        marked.update(new_refr.keys())

    return sz

def load_disk_dirs(params, feature_generator, extractor):
    """
    Computes directories according to train/dev filenames and features, for disk serialization.
    :param params: json params
    :param feature_generator: nlplingo feature generator
    :param extractor: nlplingo.nn.extractor
    :return: directory to output training examples, directory to output dev
        examples, directory to output test examples.  Each may be None.
    """

    disk_base_path = extractor.extractor_params['disk_base_path']
    feature_result = ''
    for st in feature_generator.features.feature_strings:
        feature_result += st

    def get_dir(split):
        if split not in params['data']:
            return
        normalized = params['data'][split]['filelist'].replace('/', '_').replace('.', '_')
        return os.path.join(disk_base_path, normalized, feature_result)

    train_disk_dir = get_dir("train")
    dev_disk_dir = get_dir("dev")
    test_disk_dir = get_dir("test")

    return train_disk_dir, dev_disk_dir, test_disk_dir

def get_test_examples_labels(dev_example_list, batch_size):
    """
    :param dev_example_list: list of filenames containing dev examples
    :param batch_size: int
    :return: list of nlplingo dev examples, dev labels
    """
    dev_chunk_generator = divide_chunks(dev_example_list, NUM_BIG_CHUNKS)
    test_examples = []

    # dev_chunk_generator yields lists, each of len == NUM_BIG_CHUNKS
    for big_chunk in dev_chunk_generator:
        chunk_lst = load_big_chunk(big_chunk)   # big_chunk is a filepath to .npz
        example_lst = []
        for chunk in chunk_lst:
            example_lst.extend(chunk)

        example_generator = divide_chunks(example_lst, batch_size)
        for example_chunk in example_generator:
            test_examples.extend(example_chunk)

    labels = [example.label for example in test_examples]
    test_label = np.asarray(labels)

    return test_examples, test_label
