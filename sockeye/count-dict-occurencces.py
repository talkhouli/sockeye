#!/usr/bin/python
import logging
import numpy
import os
import gzip
import argparse
import pprint
import data_io

logger = logging.getLogger(__name__)


def count_occurences(config,dictionary):
    # invert mapping
    tgt_src_dictionary = {}
    dictionary_size = 0
    for seq_idx,line in dictionary.items():
        tgt_src_line = {v: k for k, v in  line.items()}
        tgt_src_dictionary[seq_idx] = tgt_src_line
        dictionary_size += len(line.items())
    
	# Read file 
    with open(config['input_file']) as fin:
        lines = fin.readlines()
    seq_idx = 0
    unique_count = 0
    count = 0
    for line in lines:
        words = line.split( )
        newline=''
        used_words = dict()
        for word in words:
            if word in tgt_src_dictionary[seq_idx]:
                count += 1
                if word not in used_words:
                    unique_count +=1
                    used_words[word] = 1
        seq_idx += 1

    print("count of dictionary target words in file (unique per sentencet): %d"%unique_count)
    print("count of dictionary target words in file: %d"%count)
    print("dictionary size: %d"%dictionary_size)
    return



def parse_args():
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dictionary", help="dictionary")
    parser.add_argument(
        "--input-file", help="Input hypothesis file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configuration = {}
    configuration['input_file'] = args.input_file
    configuration['dictionary'] = args.dictionary
    dictionary = data_io.read_dictionary(configuration['dictionary'])
    count_occurences(configuration,dictionary)
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))



if __name__ == "__main__":
    main()

