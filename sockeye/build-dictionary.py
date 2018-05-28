#!/user/bin/python
import logging
import os
import gzip
import argparse
import pprint
import data_io
import constants as C

def build_dictionary(source, translation, reference, alignment, source_stop_list, target_stop_list, ignore_subwords):
    '''

    Generate dictionary of reference words not in the translation using the alignment
    :param source: source generator
    :param translation: translation generator
    :param reference: reference generator
    :param alignment: alignment reference-to-source generator
    :return: dictioanry of "source_word target_word" format
    '''
    dictionary = []
    for source_seq, translation_seq, reference_seq, alignment_seq in zip(source,translation,reference,alignment):
        dictionary_for_seq = {}
        #keep track of multialigned source words to avoid using them, only singly-aligned words are used
        alignment_s2t_count = [0] * (max(alignment_seq)+1)
        for j in alignment_seq:
            alignment_s2t_count[j]+=1

        for ref_idx,ref_word in enumerate(reference_seq):
            source_word = source_seq[alignment_seq[ref_idx]]
            if ref_word not in translation_seq and \
                    alignment_s2t_count[alignment_seq[ref_idx]]==1 and \
                    ref_word not in dictionary_for_seq and \
                    ref_word != C.NUM_SYMBOL and \
                    ref_word != C.UNK_SYMBOL and \
                            source_word != C.NUM_SYMBOL and \
                    (source_stop_list is None or source_word.lower() not in source_stop_list) and \
                    (target_stop_list is None or ref_word.lower() not in target_stop_list) and \
                    (not ignore_subwords or (not ref_word.endswith(C.SUBWORD_SUFFIX) and
                                                 not source_word.endswith(C.SUBWORD_SUFFIX))):
                dictionary_for_seq[ref_word] = source_word
        dictionary.append(dictionary_for_seq)
    return dictionary

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
    parser.add_argument("--ignore-subwords",
                        action="store_true",
                        help="ignore entries with source or target subwords")
    parser.add_argument("--max-per-sequence",
                        type=int,
                        default=10,
                        help="maximum dictionary entries per sequence")

    parser.add_argument("--stop-list",
                        default=None,
                        help="source stop words file")

    parser.add_argument("--target-stop-list",
                        default=None,
                        help="target stop words file")
    return parser.parse_args()

def main():
    args= parse_args()
    configuration = {}
    configuration['reference'] = args.reference
    configuration['source'] = args.source
    configuration['alignment'] = args.alignment
    configuration['output'] = args.output
    configuration['translation'] = args.translation
    configuration['stop_list'] = args.stop_list
    configuration['target_stop_list'] = args.target_stop_list
    configuration['max_per_sequence'] = args.max_per_sequence
    configuration['ignore_subwords'] = args.ignore_subwords
    source = data_io.read_content(configuration["source"])
    translation = data_io.read_content(configuration["translation"])
    alignment = data_io.read_content(configuration["alignment"],alignment=True)
    reference = data_io.read_content(configuration["reference"])
    stop_list = data_io.read_stop_list(configuration["stop_list"]) if args.stop_list else None
    target_stop_list = data_io.read_stop_list(configuration["target_stop_list"]) if args.target_stop_list else None
    dictionary = build_dictionary(source,translation,reference,alignment,stop_list,target_stop_list,configuration['ignore_subwords'])
    data_io.print_dictionary(configuration["output"],dictionary,configuration["max_per_sequence"])

if __name__ == "__main__":
    main()
