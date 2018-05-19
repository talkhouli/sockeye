#!/user/bin/python
import logging
import os
import gzip
import argparse
import pprint
import data_io

NUM_TOKEN="$number"
def build_dictionary(source,translation,reference,alignment):
    '''

    Generate dictionary of reference words not in the translation using the alignment
    :param source: source generator
    :param translation: translation generator
    :param reference: reference generator
    :param alignment: alignment reference-to-source generator
    :return: dictioanry of "source_word target_word" format
    '''
    dictionary = {}
    for source_seq, translation_seq, reference_seq, alignment_seq in zip(source,translation,reference,alignment):
        for ref_idx,ref_word in enumerate(reference_seq):
            if ref_word not in translation_seq and \
                            ref_word not in dictionary and \
                            ref_word != NUM_TOKEN and \
                            source_seq[alignment_seq[ref_idx]] != NUM_TOKEN:
                dictionary[ref_word] = source_seq[alignment_seq[ref_idx]]
    return dictionary


def print(output_file: str,dictionary: dict):
    '''
    print dictionary in format: source_word target_word
    :param output_file
    :param dictionary: dictionary dict[target_word]=source_word
    :return:
    '''
    with open(output_file,'w') as file:
        for target_word, source_word in dictionary.items():
            file.write("%s %s\n" % (source_word,target_word))

# def read_file(file):
#     sequences = []
#     with open(file,'r') as file:
#         for line in file:
#             sequences.append(line.split())
#     return sequences

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",
                        help="reference file")
    parser.add_argument("--source",
                        help="source file")
    parser.add_argument("--translation",
                        help="translation file")
    parser.add_argument("--output",
                        help="dictionary output file")
    parser.add_argument("--alignment",
                        help="alignment file")
    return parser.parse_args()

def main():
    args= parse_args()
    configuration = {}
    configuration['reference'] = args.reference
    configuration['source'] = args.source
    configuration['alignment'] = args.alignment
    configuration['output'] = args.output
    configuration['translation'] = args.translation
    source = data_io.read_content(configuration["source"])
    translation = data_io.read_content(configuration["translation"])
    alignment = data_io.read_content(configuration["alignment"],alignment=True)
    reference = data_io.read_content(configuration["reference"])
    dictionary = build_dictionary(source,translation,reference,alignment)
    print(configuration["output"],dictionary )

if __name__ == "__main__":
    main()
