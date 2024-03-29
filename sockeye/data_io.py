# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import gzip
import logging
import math
import pickle
import random
from collections import OrderedDict
from typing import Any, Dict, Iterator, Iterable, List, NamedTuple, Optional, Tuple

import mxnet as mx
import numpy as np

from utils import check_condition
import config
import constants as C

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step=10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.
    :return: List of bucket sizes.
    """
    buckets = [bucket_len for bucket_len in range(step, max_seq_len + step, step)]
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param length_ratio: Length ratio of data (target/source).
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if length_ratio >= 1.0:
        # target side is longer -> scale source
        source_step_size = max(1, int(bucket_width / length_ratio))
    else:
        # source side is longer, -> scale target
        target_step_size = max(1, int(bucket_width * length_ratio))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # Extra buckets
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    # deduplicate for return
    return list(OrderedDict.fromkeys(parallel_buckets))


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket.
    """
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


def length_statistics(source_sentences: Iterable[List[Any]],
                      target_sentences: Iterable[List[Any]]) -> Tuple[float, float]:
    """
    Returns mean and standard deviation of target-to-source length ratios of parallel corpus.

    :param source_sentences: Source sentences.
    :param target_sentences: Target sentences.
    :return: Mean and standard deviation of length ratios.
    """
    length_ratios = np.array([len(t)/float(len(s)) for t, s in zip(target_sentences, source_sentences)])
    mean = np.asscalar(np.mean(length_ratios))
    std = np.asscalar(np.std(length_ratios))
    return mean, std


def get_training_data_iters(source: str, target: str,
                            validation_source: str, validation_target: str,
                            vocab_source: Dict[str, int], vocab_target: Dict[str, int],
                            vocab_source_path: Optional[str], vocab_target_path: Optional[str],
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            fill_up: str,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            sequence_limit: Optional[int] = None,
                            alignment: str = None,
                            validation_alignment: str = None,
                            output_type: str = C.WORDS) -> Tuple['ParallelBucketSentenceIter',
                                                                           'ParallelBucketSentenceIter',
                                                                           'DataConfig']:
    """
    Returns data iterators for training and validation data.

    :param source: Path to source training data.
    :param target: Path to target training data.
    :param validation_source: Path to source validation data.
    :param validation_target: Path to target validation data.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param vocab_source_path: Path to source vocabulary.
    :param vocab_target_path: Path to target vocabulary.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :param sequence_limit: Maximum number of training sequences to read.
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("Creating train data iterator")
    # streams id-coded sentences from disk
    train_source_sentences = SentenceReader(source, vocab_source, add_bos=output_type != C.WORDS,
                                            add_eos=True,
                                            limit=sequence_limit)
    train_target_sentences = SentenceReader(target, vocab_target, add_bos=True, limit=sequence_limit)
    train_alignment_sentences = AlignmentReader(alignment, add_bos=output_type != C.WORDS,
                                                add_eos=True,
                                                limit=sequence_limit,
                                                source_lengths=[len(s) for s in train_source_sentences]) if alignment is not None else []

    # reads the id-coded sentences from disk once
    lr_mean, lr_std = length_statistics(train_source_sentences, train_target_sentences)
    check_condition(train_source_sentences.is_done() and train_target_sentences.is_done(),
                    "Different number of lines in source and target data.")
    logger.info("%d source sentences in '%s'", train_source_sentences.count, source)
    logger.info("%d target sentences in '%s'", train_target_sentences.count, target)
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)", lr_mean, lr_std)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source,
                                      max_seq_len_target,
                                      bucket_width,
                                      lr_mean) if bucketing else [
        (max_seq_len_source, max_seq_len_target)]

    train_iter = ParallelBucketSentenceIter(train_source_sentences,
                                            train_target_sentences,
                                            buckets,
                                            batch_size,
                                            batch_by_words,
                                            batch_num_devices,
                                            vocab_target[C.EOS_SYMBOL],
                                            C.PAD_ID,
                                            vocab_target[C.UNK_SYMBOL],
                                            train_alignment_sentences,
                                            label_name=C.TARGET_LABEL_NAME if output_type == C.WORDS else C.ALIGNMENT_JUMP_LABEL_NAME,
                                            bucket_batch_sizes=None,
                                            fill_up=fill_up,
                                            output_type=output_type)

    logger.info("Creating validation data iterator")
    val_source_sentences = SentenceReader(validation_source, vocab_source, add_bos=output_type !=  C.WORDS,
                                          add_eos=True,
                                          limit=None)
    val_target_sentences = SentenceReader(validation_target, vocab_target, add_bos=True, limit=None)
    val_alignment_sentences = AlignmentReader(validation_alignment, add_bos=output_type != C.WORDS,
                                              add_eos=True,
                                              limit=sequence_limit,
                                              source_lengths=[len(s) for s in val_source_sentences]) if alignment is not None else []

    val_iter = ParallelBucketSentenceIter(val_source_sentences,
                                          val_target_sentences,
                                          buckets,
                                          batch_size,
                                          batch_by_words,
                                          batch_num_devices,
                                          vocab_target[C.EOS_SYMBOL],
                                          C.PAD_ID,
                                          vocab_target[C.UNK_SYMBOL],
                                          val_alignment_sentences,
                                          label_name=C.TARGET_LABEL_NAME if output_type == C.WORDS else C.ALIGNMENT_JUMP_LABEL_NAME,
                                          bucket_batch_sizes=train_iter.bucket_batch_sizes,
                                          fill_up=fill_up,
                                          output_type=output_type)

    check_condition(val_source_sentences.is_done() and val_target_sentences.is_done(),
                    "Different number of lines in source and target validation data.")
    logger.info("%d validation source sentences in '%s'", val_source_sentences.count, source)
    logger.info("%d validation target sentences in '%s'", val_target_sentences.count, target)

    config_data = DataConfig(source, target,
                             validation_source, validation_target,
                             vocab_source_path, vocab_target_path,
                             lr_mean, lr_std, train_iter.max_observed_source_len, train_iter.max_observed_target_len,
                             alignment,validation_alignment)

    return train_iter, val_iter, config_data


class DataConfig(config.Config):
    """
    Stores data paths from training.
    """
    def __init__(self,
                 source: str,
                 target: str,
                 validation_source: str,
                 validation_target: str,
                 vocab_source: Optional[str],
                 vocab_target: Optional[str],
                 length_ratio_mean: float = C.TARGET_MAX_LENGTH_FACTOR,
                 length_ratio_std: float = 0.0,
                 max_observed_source_seq_len: Optional[int] = None,
                 max_observed_target_seq_len: Optional[int] = None,
                 alignment: str = None,
                 validation_alignment :str = None) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.alignment = alignment
        self.validation_source = validation_source
        self.validation_target = validation_target
        self.validation_alignment = validation_alignment
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.max_observed_source_seq_len = max_observed_source_seq_len
        self.max_observed_target_seq_len = max_observed_target_seq_len


def smart_open(filename: str, mode: str = "rt", ftype: str = "auto", errors:str = 'replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open
    :param errors: Encoding error handling during reading. Defaults to 'replace'
    :return: File descriptor
    """
    if ftype == 'gzip' or ftype == 'gz' or (ftype == 'auto' and filename.endswith(".gz")):
        return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)


def read_content(path: str, limit: Optional[int] = None, alignment: bool = False) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of words.
    """
    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            if alignment:
                align_pairs = get_alignment(line)
                align = [-1] * max([i+1 for _,i in align_pairs])
                for j,i in get_alignment(line):
                    assert align[i] == -1
                    align[i] = j
                yield align
            else:
                yield list(get_tokens(line))

def print_dictionary(output_file: str,dictionary: [dict],max_per_sequence):
    '''
    print dictionary in format: source_word target_word
    :param output_file
    :param dictionary: list of dictionaries dict[seq_idx][target_word]=source_word
    :param max_per_sequence: maximum number of dictionary entries per sequence
    :return:
    '''
    with open(output_file,'w') as file:
        for seq_dictionary in dictionary:
            count =0
            for target_word, source_word in seq_dictionary.items():
                file.write("%s\t%s #" % (source_word,target_word))
                count += 1
                if count >= max_per_sequence:
                    break
            file.write("\n")

def read_stop_list(file):
    '''
    reads stop list file consisting of one word per line
    :param file: path to stop list file
    :return: stop_list: list of stop words
    '''
    stop_list = []
    with open(file,'r') as file:
        for line in file:
            stop_list.append(line.strip().lower())
    return stop_list


def read_dictionary(dictionary_file):
    '''
    read dictionary file
    :param dictionary_file: word-to-word dictionary file, with the format src_word target_word
    :return: dict(source_word)->target_word
    '''
    with open(dictionary_file,'r') as file:
        dictionary = {}
        for seq_idx,line in enumerate(file):
            dictionary[seq_idx] = dict()
            entries = line.split("#")
            for entry in entries:
                toks = entry.split()
                if len(toks)==2:
                    dictionary[seq_idx][toks[0]]= toks[1]
    return dictionary

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def get_alignment(line: str) -> (int,int):
    """
    Yields alignments pairs (src_pos, tgt_pos)

    :param line: Alignment input string.
    :return: Iterator over alignment points.
    """
    done = False
    src_pos = -1
    tgt_pos = -1
    cnt = -1
    for token in line.rstrip().split():
        cnt+=1
        if token == "S" and cnt == 0:
            continue
        if cnt == 1:
            src_pos = int(token)
        elif cnt == 2:
            tgt_pos = int(token)
            cnt=-1
            yield (src_pos,tgt_pos)



def tokens2ids(tokens: Iterable[str], vocab: Dict[str, int], 
                has_categ_content= False) -> List[int]:
    """
    Returns sequence of ids given a sequence of tokens and vocab.

    :param tokens: List of tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :param has_categ_content: set to true during inference, correctly map words
                                prefixed with C.NUM_SYMBOL + '_'
    :return: List of word ids.
    """
    return [vocab.get(C.NUM_SYMBOL, vocab[C.UNK_SYMBOL])
                if w.startswith(C.NUM_SYMBOL + '_')
                else vocab.get(C.NUM_SYMBOL_2, vocab[C.UNK_SYMBOL])
                    if w.startswith(C.NUM_SYMBOL_2 + '_')
                    else vocab.get(w, vocab[C.UNK_SYMBOL])
            for w in tokens]


class SentenceReader(Iterator):
    """
    Reads sentences from path and creates word id sentences.
    Streams from disk, instead of loading all sentences into memory.

    :param path: Path to read data from.
    :param vocab: Vocabulary mapping.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    """

    def __init__(self, path: str, vocab: Dict[str, int], add_bos: bool = False,
                 add_eos:bool = False, limit: Optional[int] = None) -> None:
        self.path = path
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos=add_eos
        self.limit = limit
        assert C.UNK_SYMBOL in vocab
        assert C.UNK_SYMBOL in vocab
        assert vocab[C.PAD_SYMBOL] == C.PAD_ID
        assert C.BOS_SYMBOL in vocab
        assert C.EOS_SYMBOL in vocab
        self._iter = None  # type: Optional[Iterator]
        self._iterated_once = False
        self.count = 0
        self._next = None

    def __iter__(self):
        assert self._next is None, "Can not iterate multiple times simultaneously."
        self._iter = read_content(self.path, self.limit)
        self._next = next(self._iter, None)
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration

        sentence_tokens = self._next
        sentence = tokens2ids(sentence_tokens, self.vocab)
        check_condition(bool(sentence), "Empty sentence in file %s" % self.path)
        if self.add_bos:
            sentence.insert(0, self.vocab[C.BOS_SYMBOL])

        if self.add_eos:
            sentence.append(self.vocab[C.EOS_SYMBOL])

        if not self._iterated_once:
            self.count += 1

        # fetch next element
        self._next = next(self._iter, None)
        if self._next is None:
            self._iter = None
            if not self._iterated_once:
                self._iterated_once = True

        return sentence

    def is_done(self):
        return self._iterated_once and self._next is None

class AlignmentReader(Iterator):
    """
    Reads sentences from path and creates word id sentences.
    Streams from disk, instead of loading all sentences into memory.

    :param path: Path to read data from.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    """

    def __init__(self, path: str, add_bos: bool = False, add_eos:bool = False,
                 limit: Optional[int] = None, source_lengths: List[int] = None) -> None:
        self.path = path
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.limit = limit
        self._iter = None  # type: Optional[Iterator]
        self._iterated_once = False
        self.count = 0
        self._next = None
        self.source_lengths = source_lengths
        self.last_idx = 0

    def __iter__(self):
        assert self._next is None, "Can not iterate multiple times simultaneously."
        self._iter = read_content(self.path, self.limit, alignment=True)
        self._next = next(self._iter, None)
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration

        alignment = self._next
        check_condition(bool(alignment), "Empty alignment in file %s" % self.path)
        if self.add_bos:
            alignment = [i+1 if i != -1 else -1 for i in alignment]
            alignment.insert(0,0)

        if self.add_eos:
            alignment.append(self.source_lengths[self.last_idx]-1)

        self.last_idx += 1

        if not self._iterated_once:
            self.count += 1

        # fetch next element
        self._next = next(self._iter, None)
        if self._next is None:
            self._iter = None
            if not self._iterated_once:
                self._iterated_once = True

        return alignment

    def is_done(self):
        return self._iterated_once and self._next is None


def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)


def get_parallel_bucket(buckets: List[Tuple[int, int]],
                        length_source: int,
                        length_target: int) -> Optional[Tuple[int, Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None  # type: Tuple[int, Tuple[int, int]]
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            bucket = j, (source_bkt, target_bkt)
            break
    return bucket


BucketBatchSize = NamedTuple("BucketBatchSize", [
    ("batch_size", int),
    ("average_words_per_batch", float)
])
"""
:param batch_size: Number of sentences in each batch.
:param average_words_per_batch: Approximate number of non-padding tokens in each batch.
"""


# TODO: consider more memory-efficient batch creation (load from disk on demand)
# TODO: consider using HDF5 format for language data
class ParallelBucketSentenceIter(mx.io.DataIter):
    """
    A bucketing parallel sentence iterator.
    Data is read into NDArrays for the buckets defined in buckets.
    Randomly shuffles the data after every call to reset().
    Data is stored in NDArrays for each epoch for fast indexing during iteration.

    :param source_sentences: Iterable of source sentences (integer-coded).
    :param target_sentences: Iterable of target sentences (integer-coded).
    :param buckets: List of buckets.
    :param batch_size: Batch_size of generated data batches.
           Incomplete batches are discarded if fill_up == None, or filled up according to the fill_up strategy.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: If not None, fill up bucket data to a multiple of batch_size to avoid discarding incomplete batches.
           for each bucket. If set to 'replicate', sample examples from the bucket and use them to fill up.
    :param eos_id: Word id for end-of-sentence.
    :param pad_id: Word id for padding symbols.
    :param unk_id: Word id for unknown symbols.
    :param bucket_batch_sizes: Pre-computed bucket batch sizes (used to keep iterators consistent for train/validation).
    :param dtype: Data type of generated NDArrays.
    """

    def __init__(self,
                 source_sentences: Iterable[List[int]],
                 target_sentences: Iterable[List[int]],
                 buckets: List[Tuple[int, int]],
                 batch_size: int,
                 batch_by_words: bool,
                 batch_num_devices: int,
                 eos_id: int,
                 pad_id: int,
                 unk_id: int,
                 alignment: Iterable[List[int]] = [],
                 bucket_batch_sizes: Optional[List[BucketBatchSize]] = None,
                 fill_up: Optional[str] = None,
                 output_type: str = C.WORDS,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 alignment_data_name=C.ALIGNMENT_NAME,
                 last_alignment_data_name=C.LAST_ALIGNMENT_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 dtype='float32') -> None:
        super(ParallelBucketSentenceIter, self).__init__()

        self.buckets = list(buckets)
        self.buckets.sort()
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.batch_size = batch_size
        self.batch_by_words = batch_by_words
        self.batch_num_devices = batch_num_devices
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.dtype = dtype
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.alignment_data_name = alignment_data_name
        self.last_alignment_data_name = last_alignment_data_name
        self.label_name = label_name
        self.fill_up = fill_up
        self.output_type=output_type

        self.data_source = [[] for _ in self.buckets]  # type: ignore
        self.data_target = [[] for _ in self.buckets]  # type: ignore
        self.data_label = [[] for _ in self.buckets]  # type: ignore
        self.data_alignment = None
        self.data_last_alignment = None
        if alignment != []:
            self.data_alignment = [[] for _ in self.buckets]  # type: ignore
            self.data_last_alignment = [[] for _ in self.buckets]  # type: ignore

        self.data_target_average_len = [0 for _ in self.buckets]

        # Per-bucket batch sizes (num seq, num word)
        # If not None, populated as part of assigning to buckets
        self.bucket_batch_sizes = bucket_batch_sizes

        # assign sentence pairs to buckets
        self.max_observed_source_len = 0
        self.max_observed_target_len = 0
        self._assign_to_buckets(source_sentences, target_sentences, alignment)

        # convert to single numpy array for each bucket
        self._convert_to_array()

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        self.provide_data = [
            mx.io.DataDesc(name=self.source_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[0]),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=self.target_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=self.label_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.target_data_name]
        self.label_names = [self.label_name]

        if alignment != []:
            self.provide_data.append(
                                mx.io.DataDesc(name=self.alignment_data_name,
                                shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                                layout=C.BATCH_MAJOR))
            self.data_names.append(self.alignment_data_name)

            # TODO only provide data if it is actually used
            # remove debug symbol last alignment if that is solved
            self.provide_data.append(
                mx.io.DataDesc(name=self.last_alignment_data_name,
                               shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                               layout=C.BATCH_MAJOR))

            self.data_names.append(self.last_alignment_data_name)

        # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
        self.idx = []  # type: List[Tuple[int, int]]
        for i, buck in enumerate(self.data_source):
            batch_size_seq = self.bucket_batch_sizes[i].batch_size
            rest = len(buck) % batch_size_seq
            if rest > 0:
                logger.info("Discarding %d samples from bucket %s due to incomplete batch", rest, self.buckets[i])
            idxs = [(i, j) for j in range(0, len(buck) - batch_size_seq + 1, batch_size_seq)]
            self.idx.extend(idxs)
        self.curr_idx = 0

        # holds NDArrays
        self.indices = []  # type: List[List[int]]
        self.nd_source = []  # type: List[mx.ndarray]
        self.nd_target = []  # type: List[mx.ndarray]
        self.nd_label = []  # type: List[mx.ndarray]

        self.nd_alignment = []  if alignment != [] else None # type: List[mx.ndarray]
        self.nd_last_alignment = [] if alignment != [] else None  # type: List[mx.ndarray]

        self.reset()

    def _assign_to_buckets(self, source_sentences, target_sentences, alignment_sentences):
        ndiscard = 0
        tokens_source = 0
        tokens_target = 0
        num_of_unks_source = 0
        num_of_unks_target = 0

        # Bucket sentences as padded np arrays
        zip_list = [source_sentences, target_sentences] + ([alignment_sentences] if alignment_sentences!=[] else [])
        for elements in zip(*zip_list):
            source = elements[0]
            target = elements[1]
            alignment = elements[2] if len(elements)>2 else None
            # last aligned positions: unaligned target words mapped to last aligned source positions
            if alignment:
                last_alignment = alignment[:]
                for i in range(0, len(target)):
                    if last_alignment[i] == C.UNALIGNED_SOURCE_INDEX:
                        last_alignment[i] = last_alignment[i - 1]

            source_len = len(source)
            target_len = len(target)
            buck_idx, buck = get_parallel_bucket(self.buckets, source_len, target_len)
            if buck is None:
                ndiscard += 1
                continue  # skip this sentence pair

            tokens_source += source_len
            tokens_target += target_len
            if source_len > self.max_observed_source_len:
                self.max_observed_source_len = source_len
            if target_len > self.max_observed_target_len:
                self.max_observed_target_len = target_len

            num_of_unks_source += source.count(self.unk_id)
            num_of_unks_target += target.count(self.unk_id)

            buff_source = np.full((buck[0],), self.pad_id, dtype=self.dtype)
            buff_target = np.full((buck[1],), self.pad_id, dtype=self.dtype)
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 0.13): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            buff_source[:source_len] = source
            buff_target[:target_len] = target
            # using self.pad_id also for the alignment model. Since self.pad_id=0 means jump=-100, this should not
            # make a real difference
            buff_label = np.full((buck[1],), self.pad_id, dtype=self.dtype)
            if self.output_type == C.WORDS:
                buff_label[:len(target)] = target[1:] + [self.eos_id]
            else:
                offset = (C.NUM_ALIGNMENT_JUMPS-1)/2
                unaligned = np.array([alignment[i+1] == C.UNALIGNED_SOURCE_INDEX for i in range(0,len(target))])

                buff_label[:len(target)] = [offset + last_alignment[i] - last_alignment[i-1] for i in range(1,len(target)+1)]
                                           #+ [offset + len(source)-alignment[len(target)-1]]

                buff_label[:len(target)][unaligned] = C.UNALIGNED_JUMP_LABEL
            self.data_source[buck_idx].append(buff_source)
            self.data_target[buck_idx].append(buff_target)
            self.data_label[buck_idx].append(buff_label)
            self.data_target_average_len[buck_idx] += target_len

            if alignment is not None:
                buff_alignment = np.full((buck[1]+1,), self.pad_id, dtype=self.dtype)
                buff_alignment[:len(alignment)] = alignment
                self.data_alignment[buck_idx].append(buff_alignment)

                buff_last_alignment = np.full((buck[1] + 1,), self.pad_id, dtype=self.dtype)
                buff_last_alignment[:len(alignment)] = last_alignment
                self.data_last_alignment[buck_idx].append(buff_last_alignment)

        # Average number of non-padding elements in target sequence per bucket
        for buck_idx, buck in enumerate(self.buckets):
            # Case of empty bucket -> use default padded length
            if self.data_target_average_len[buck_idx] == 0:
                self.data_target_average_len[buck_idx] = buck[1]
            else:
                self.data_target_average_len[buck_idx] /= len(self.data_target[buck_idx])

        # We now have sufficient information to populate bucket batch sizes
        self._populate_bucket_batch_sizes()

        logger.info("Source words: %d", tokens_source)
        logger.info("Target words: %d", tokens_target)
        logger.info("Vocab coverage source: %.0f%%", (1 - num_of_unks_source / tokens_source) * 100)
        logger.info("Vocab coverage target: %.0f%%", (1 - num_of_unks_target / tokens_target) * 100)
        logger.info("Total: %d samples in %d buckets", sum(len(b) for b in self.data_source), len(self.buckets))
        nsamples = 0
        for bkt, buck, batch_size_seq, average_seq_len in zip(self.buckets,
                                                              self.data_source,
                                                              (bbs.batch_size for bbs in self.bucket_batch_sizes),
                                                              self.data_target_average_len):
            logger.info("Bucket of %s : %d samples in %d batches of %d, approx %0.1f words/batch",
                        bkt,
                        len(buck),
                        math.ceil(len(buck) / batch_size_seq),
                        batch_size_seq,
                        batch_size_seq * average_seq_len)
            nsamples += len(buck)
        check_condition(nsamples > 0, "0 data points available in the data iterator. "
                                      "%d data points have been discarded because they "
                                      "didn't fit into any bucket. Consider increasing "
                                      "--max-seq-len to fit your data." % ndiscard)
        logger.info("%d sentence pairs out of buckets", ndiscard)
        logger.info("fill up mode: %s", self.fill_up)
        logger.info("")

    def _populate_bucket_batch_sizes(self):
        """
        Compute bucket-specific batch sizes (sentences, average_words) and default bucket batch
        size.

        If sentence-based batching: number of sentences is the same for each batch, determines the
        number of words.

        If word-based batching: number of sentences for each batch is set to the multiple of number
        of devices that produces the number of words closest to the target batch size.  Average
        target sentence length (non-padding symbols) is used for word number calculations.

        Sets: self.bucket_batch_sizes
        """
        # Pre-defined bucket batch sizes
        if self.bucket_batch_sizes is not None:
            return
        # Otherwise compute here
        self.bucket_batch_sizes = [None for _ in self.buckets]
        largest_total_batch_size = 0
        for buck_idx, bucket_shape in enumerate(self.buckets):
            # Target/label length with padding
            padded_seq_len = bucket_shape[1]
            # Average target/label length excluding padding
            average_seq_len = self.data_target_average_len[buck_idx]
            # Word-based: num words determines num sentences
            # Sentence-based: num sentences determines num words
            if self.batch_by_words:
                check_condition(padded_seq_len <= self.batch_size, "Word batch size must cover sequence lengths for all"
                                " buckets: (%d > %d)" % (padded_seq_len, self.batch_size))
                # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
                # average length
                batch_size_seq = self.batch_num_devices * round((self.batch_size / average_seq_len)
                                                                / self.batch_num_devices)
                batch_size_word = batch_size_seq * average_seq_len
            else:
                batch_size_seq = self.batch_size
                batch_size_word = batch_size_seq * average_seq_len
            self.bucket_batch_sizes[buck_idx] = BucketBatchSize(batch_size_seq, batch_size_word)
            # Track largest batch size by total elements
            largest_total_batch_size = max(largest_total_batch_size, batch_size_seq * max(*bucket_shape))
        # Final step: guarantee that largest bucket by sequence length also has largest total batch size.
        # When batching by sentences, this will already be the case.
        if self.batch_by_words:
            padded_seq_len = max(*self.buckets[-1])
            average_seq_len = self.data_target_average_len[-1]
            while self.bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_batch_size:
                self.bucket_batch_sizes[-1] = BucketBatchSize(
                    self.bucket_batch_sizes[-1].batch_size + self.batch_num_devices,
                    self.bucket_batch_sizes[-1].average_words_per_batch + self.batch_num_devices * average_seq_len)

    def _convert_to_array(self):
        for i in range(len(self.data_source)):
            self.data_source[i] = np.asarray(self.data_source[i], dtype=self.dtype)
            self.data_target[i] = np.asarray(self.data_target[i], dtype=self.dtype)
            self.data_label[i] = np.asarray(self.data_label[i], dtype=self.dtype)
            if self.data_alignment is not None:
                self.data_alignment[i] = np.asarray(self.data_alignment[i], dtype=self.dtype)

            if self.data_last_alignment is not None:
                self.data_last_alignment[i] = np.asarray(self.data_last_alignment[i], dtype=self.dtype)


            n = len(self.data_source[i])
            batch_size_seq = self.bucket_batch_sizes[i].batch_size
            if n % batch_size_seq != 0:
                buck_shape = self.buckets[i]
                rest = batch_size_seq - n % batch_size_seq
                if self.fill_up == 'pad':
                    raise NotImplementedError
                elif self.fill_up == 'replicate':
                    logger.info("Replicating %d random sentences from bucket %s to size it to multiple of %d", rest,
                                buck_shape, batch_size_seq)
                    random_indices = np.random.randint(self.data_source[i].shape[0], size=rest)
                    self.data_source[i] = np.concatenate((self.data_source[i], self.data_source[i][random_indices, :]),
                                                         axis=0)
                    self.data_target[i] = np.concatenate((self.data_target[i], self.data_target[i][random_indices, :]),
                                                         axis=0)
                    self.data_label[i] = np.concatenate((self.data_label[i], self.data_label[i][random_indices, :]),
                                                         axis=0)
                    if self.data_alignment is not None:
                        self.data_alignment[i] = np.concatenate((self.data_alignment[i],
                                                                 self.data_alignment[i][random_indices, :]), axis=0)
                    if self.data_last_alignment is not None:
                        self.data_last_alignment[i] = np.concatenate((self.data_last_alignment[i],
                                                                 self.data_last_alignment[i][random_indices, :]), axis=0)


    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_idx = 0
        # shuffle indices
        random.shuffle(self.idx)

        self.nd_source = []
        self.nd_target = []
        self.nd_label = []
        if self.nd_alignment is not None:
            self.nd_alignment = []

        if self.nd_last_alignment is not None:
            self.nd_last_alignment = []

        self.indices = []
        for i in range(len(self.data_source)):
            # shuffle indices within each bucket
            self.indices.append(np.random.permutation(len(self.data_source[i])))
            self._append_ndarrays(i, self.indices[-1])

    def _append_ndarrays(self, bucket: int, shuffled_indices: np.array):
        """
        Appends the actual data, selected by the given indices, to the NDArrays
        of the appropriate bucket. Use when reshuffling the data.

        :param bucket: Current bucket.
        :param shuffled_indices: Indices indicating which data to select.
        """
        self.nd_source.append(mx.nd.array(self.data_source[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_target.append(mx.nd.array(self.data_target[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_label.append(mx.nd.array(self.data_label[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        if self.nd_alignment is not None:
            self.nd_alignment.append(mx.nd.array(self.data_alignment[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))

        if self.nd_last_alignment is not None:
            self.nd_last_alignment.append(mx.nd.array(self.data_last_alignment[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_idx != len(self.idx)

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        batch_size_seq = self.bucket_batch_sizes[i].batch_size
        source = self.nd_source[i][j:j + batch_size_seq]
        target = self.nd_target[i][j:j + batch_size_seq]
        data = [source, target]

        if self.nd_alignment is not None:
            alignment = self.nd_alignment[i][j:j + batch_size_seq]
            data.append(alignment)

        if self.nd_last_alignment is not None:
            last_alignment = self.nd_last_alignment[i][j:j + batch_size_seq]
            data.append(last_alignment)

        label = [self.nd_label[i][j:j + batch_size_seq]]

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        # TODO: num pad examples is not set here if fillup strategy would be padding
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.idx, fp)
            pickle.dump(self.curr_idx, fp)
            np.save(fp, self.indices)

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """
        with open(fname, "rb") as fp:
            self.idx = pickle.load(fp)
            self.curr_idx = pickle.load(fp)
            self.indices = np.load(fp)

        # Because of how checkpointing is done (pre-fetching the next batch in
        # each iteration), curr_idx should be always >= 1
        assert self.curr_idx >= 1
        # Right after loading the iterator state, next() should be called
        self.curr_idx -= 1

        self.nd_source = []
        self.nd_target = []
        self.nd_label = []
        if self.nd_alignment is not None:
            self.nd_alignment = []

        if self.nd_last_alignment is not None:
            self.nd_last_alignment = []

        for i in range(len(self.data_source)):
            self._append_ndarrays(i, self.indices[i])
