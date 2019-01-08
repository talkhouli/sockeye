#!/user/bin/python
import logging
import json
import os
import gzip
import argparse
import pprint
import data_io
import constants as C

NER = {"PERSON", "COUNTRY", "CITY", "ORGANIZATION", "LOCATION"}
def build_dictionary(source, translation, reference, alignment,
        source_stop_list, target_stop_list, ignore_subwords,
        min_src_word_length):
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
                    len(source_word)>= min_src_word_length and \
                    (not ignore_subwords or (not ref_word.endswith(C.SUBWORD_SUFFIX) and
                                                 not source_word.endswith(C.SUBWORD_SUFFIX))):
                dictionary_for_seq[ref_word] = source_word
        dictionary.append(dictionary_for_seq)
    return dictionary

def build_ibm1_dictionary(source, translation, source_stop_list,
        target_stop_list,ignore_subwords,
        word_dictionary,min_src_word_length,source_counts,
        min_source_count,
        max_source_count,
        ner):
    '''

    Generate dictionary according to IBM-1 top scoring word, do not include
    words already in the translation
    :param source: source generator
    :param source_stop_list
    :param target_stop_list
    :param ignore_subwords: do not include words with SUBWORD_SUFFIX
    :word_dictionary source-target word translations
    :return: dictioanry of "source_word target_word" format
    '''
    dictionary = []
    if  translation is not None:
        lists = (source,translation)
    else:
        lists = (source,)
    for seq_idx,pair in enumerate(zip(*lists)):
        source_seq = pair[0]
        translation_seq = pair[1] if len(pair)== 2 else None
        dictionary_for_seq = {}
        #keep track of multialigned source words to avoid using them, only singly-aligned words are used
        for word_idx,source_word in enumerate(source_seq):
            target_word = word_dictionary[source_word] if source_word in word_dictionary else None 
            if  target_word is not None and \
                    target_word != "" and \
                    target_word not in dictionary_for_seq and \
                    target_word != C.NUM_SYMBOL and \
                    target_word != C.UNK_SYMBOL and \
                    source_word != C.NUM_SYMBOL and \
                    (translation_seq is None or target_word not in
                            translation_seq)  and \
                    (ner is None or ner[seq_idx][word_idx] in NER) and \
                    len(source_word)>= min_src_word_length and \
                    source_word in source_counts and source_counts[source_word] >= min_source_count  and \
                            source_counts[source_word] <= max_source_count and \
                    (source_stop_list is None or source_word.lower() not in source_stop_list) and \
                    (target_stop_list is None or target_word.lower() not in target_stop_list) and \
                    (not ignore_subwords or (not target_word.endswith(C.SUBWORD_SUFFIX) and
                                                 not source_word.endswith(C.SUBWORD_SUFFIX))):
                print("count: %d"%source_counts[source_word])
                if ner:
                    print("ner %s"%ner[seq_idx][word_idx])
                dictionary_for_seq[target_word] = source_word
        dictionary.append(dictionary_for_seq)
    return dictionary

def create_ibm1_dict(ibm1File):

    """
    This method loads IBM-1 model 
    """

    dictionary = dict()

    wordSource = ''
    bestTargetword = ''
    score = 0
    count = 0

    with gzip.open(ibm1File, 'rt', encoding="utf-8") as file:
        for line in file:
            count += 1
            if count % 100000 == 0:
                print(count)
            words = line.split(" ")
            currentWord = words[1].strip()
            if wordSource == '':
                wordSource = currentWord
            if currentWord != wordSource:
                if wordSource is not None:
                        dictionary.update({wordSource:bestTargetword})
                wordSource = currentWord
                score = 0
                bestTargetword = ''
            if float(words[0].strip()) > score:
                score = float(words[0].strip())
                bestTargetword = words[2].strip()
            else:
                continue
    return dictionary

def read_json(json_file):
    with open(json_file) as f:
            data = json.load(f)
    return data

def read_ner(ner_file):
    data = read_json(ner_file)
    ner_data = []
    for sentence in data["sentences"]:
        ner_seq =[]
        for token in sentence["tokens"]:
            ner_seq.append(token["ner"])
            print(token["ner"])
        ner_data.append(ner_seq)
    return ner_data

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
    parser.add_argument("--min-word-length",
                        default=1,
                        type=int,
                        help="minimum source word length")
    parser.add_argument("--source-counts-file",
                        help="json file of source word counts")
    parser.add_argument("--max-per-sequence",
                        type=int,
                        default=10,
                        help="maximum dictionary entries per sequence")
    parser.add_argument("--min-source-count",
                        type=int,
                        default=0,
                        help="minimum source word count in training data")
    parser.add_argument("--max-source-count",
                        type=int,
                        default=1e10,
                        help="maximum source word count in training data")

    parser.add_argument("--stop-list",
                        default=None,
                        help="source stop words file")

    parser.add_argument("--ner",
                        default=None,
                        help="json NER file")

    parser.add_argument("--target-stop-list",
                        default=None,
                        help="target stop words file")
    
    parser.add_argument("--ibm1",
                        help="IBM-1 model")
    
    return parser.parse_args()

def main():
    args= parse_args()
    configuration = {}
    configuration['reference'] = args.reference
    configuration['source'] = args.source
    configuration['alignment'] = args.alignment
    configuration['output'] = args.output
    configuration['translation'] = args.translation
    configuration['min_word_length'] = args.min_word_length
    configuration['stop_list'] = args.stop_list
    configuration['target_stop_list'] = args.target_stop_list
    configuration['max_per_sequence'] = args.max_per_sequence
    configuration['ignore_subwords'] = args.ignore_subwords
    configuration['ibm1'] = args.ibm1
    configuration['source_counts_file'] = args.source_counts_file
    configuration['min_source_count'] = args.min_source_count
    configuration['max_source_count'] = args.max_source_count
    configuration['ner'] = args.ner
    source = data_io.read_content(configuration["source"])
    translation = data_io.read_content(configuration["translation"]) if    args.translation is not None else None
    alignment = data_io.read_content(configuration["alignment"],alignment=True)    if args.alignment is not None else None
    reference = data_io.read_content(configuration["reference"]) if    args.reference is not None else None
    stop_list = data_io.read_stop_list(configuration["stop_list"]) if args.stop_list else None
    target_stop_list = data_io.read_stop_list(configuration["target_stop_list"]) if args.target_stop_list else None
    source_counts = read_json(configuration['source_counts_file'])
    ner = read_ner(configuration['ner']) if args.ner else None
    if args.ibm1:
        word_dictionary = create_ibm1_dict(args.ibm1)
        dictionary = build_ibm1_dictionary(source,
                                translation,
                                stop_list,
                                target_stop_list,
                                configuration['ignore_subwords'],
                                word_dictionary,
                                configuration['min_word_length'],
                                source_counts,
                                configuration['min_source_count'],
                                configuration['max_source_count'],
                                ner)
    else:
        dictionary = build_dictionary(source,translation,reference,alignment,stop_list,target_stop_list,configuration['ignore_subwords'],configuration['min_word_length'])
    data_io.print_dictionary(configuration["output"],dictionary,configuration["max_per_sequence"])

if __name__ == "__main__":
    main()
