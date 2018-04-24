import argparse
import sys
from log import setup_main_logger
from utils import smart_open, check_condition
from arguments import regular_file

logger = setup_main_logger(__name__, file_logging=False)

UNALIGNED_TARGET = {
    "bbn",
    "sb",
    "eps",
    "keep"
}

MULTIPLY_ALIGNED_TARGET = {
    "bbn",
    "keep"
}

OUTPUT_FORMAT = {
    "flat",
    "multiline"
}

def is_int(s):
    """
    check if s is of type int
    :param s: value to check
    :return: return true if s is int
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_sentences(path):
    """
    read file line by line and split words
    :param path: file to read
    :return: array of lines
    """
    file = smart_open(path)
    sentences = []
    for line in file.readlines():
        sentences.append(line.strip().split(" "))

    return sentences


def _read_flat_alignment_file(content, trg_lengths):
    alignments = [] * len(trg_lengths)
    for l, line in enumerate(content):
        alignment = [-1] * (trg_lengths[l])
        source = -1

        for i, token in enumerate(line.strip().split(" ")):
            if i % 3 == 0:
                check_condition(token == "S", "Wrong alignment format: expected S 0 0. Got token %s" % token)
                continue

            if i % 3 == 1:
                check_condition(is_int(token), "expected int")
                source = int(token)

            if i % 3 == 2:
                check_condition(is_int(token), "expected int")
                target = int(token)
                check_condition(target < len(alignment),
                                "Alignment must point in to the sentence. Got alignment %d for length %d in sentence %d"
                                % (target, (trg_lengths[l]), l))
                if alignment[target] == -1:
                    alignment[target] = source
                elif isinstance(alignment[target], list):
                    alignment[target].append(source)
                else:
                    alignment[target] = [alignment[target], source]

        alignments.append(alignment)

    check_condition(len(alignments) == len(trg_lengths), "alignment mst be parallel")
    return alignments


def _read_multiline_alignment_file(content, trg_lengths):
    alignments = []
    alignment = []
    sentence = -1
    for l, line in enumerate(content):
        if line.startswith("SENT: "):
            token = line[6:]
            check_condition(is_int(token), "expected int for sentence number")
            sentence = int(token)
            check_condition(sentence < len(trg_lengths), "alignment mst be parallel")
            if len(alignment) > 0:
                alignments.append(alignment)

            alignment = [-1]*trg_lengths[sentence]
        elif line.startswith("S"):
            for i, token in enumerate(line.strip().split(" ")):
                if i % 3 == 0:
                    check_condition(token == "S", "Wrong alignment format: expected S 0 0. Got token %s" % token)
                    continue

                if i % 3 == 1:
                    check_condition(is_int(token), "expected int")
                    source = int(token)

                if i % 3 == 2:
                    check_condition(is_int(token), "expected int")
                    target = int(token)
                    check_condition(target < len(alignment),
                                    "Alignment must point in to the sentence. "
                                    "Got alignment %d for length %d in sentence %d"
                                    % (target, trg_lengths[sentence], sentence))

                    if alignment[target] == -1:
                        alignment[target] = source
                    elif isinstance(alignment[target], list):
                        alignment[target].append(source)
                    else:
                        alignment[target] = [alignment[target], source]
    alignments.append(alignment)
    check_condition(len(alignments) == len(trg_lengths), "alignment mst be parallel. alignment %d vs target %d"
                    % (len(alignments), len(trg_lengths)))
    return alignments


def read_alignment_file(path, trg_lengths, src_lengths):
    """
    read flat alignment file
    :param path: path to alignment file
    :param trg_lengths: array of target lengths (for each sentence)
    :param src_lengths: array of source lengths (for each sentence)
    :return: array of alignments (unprocessed)
    """
    check_condition(len(trg_lengths) == len(src_lengths), "source and target sentences must be parallel")
    file = smart_open(path)
    content = file.readlines()
    if len(content) == len(trg_lengths):
        is_multiline = False
        alignments = _read_flat_alignment_file(content=content, trg_lengths=trg_lengths)
    else:
        is_multiline = True
        alignments = _read_multiline_alignment_file(content=content, trg_lengths=trg_lengths)

    check_condition(len(alignments) == len(trg_lengths), "alignment mst be parallel")
    return alignments, is_multiline


def process_alignments(alignments,
                       unaligned_target,
                       multiply_aligned_target,
                       eps_index):
    """
    process alignments according to heuristics given
    :param alignments: array of alignments
    :param unaligned_target: how to handle unaligned target positions (bbn, sb, eps, keep)
    :param multiply_aligned_target:  how to handle multiply aligned target positions (bbn, keep)
    :param eps_index: index used in case of eps-unaligned heuristic
    :return: array of processed alignments
    """
    check_condition(unaligned_target in UNALIGNED_TARGET,
                    "unaligned target must be one of %s given '%s'" % (list(UNALIGNED_TARGET), unaligned_target))
    check_condition(unaligned_target != "eps" or eps_index is not None,
                    "unaligned target mode 'eps' requires --unaligned-target-epsilon-index")
    check_condition(multiply_aligned_target in MULTIPLY_ALIGNED_TARGET,
                    "multiply aligned target must be one of %s given '%s'"
                    % (list(MULTIPLY_ALIGNED_TARGET), multiply_aligned_target))

    for l in range(len(alignments)):
        for t in range(len(alignments[l])):
            # Process unaligned Target
            if alignments[l][t] == -1:
                if unaligned_target == "sb":
                    alignments[l][t] = 0
                elif unaligned_target == "bbn":
                    i = 0
                    while alignments[l][t] == -1:
                        if t + i < len(alignments[l]) and alignments[l][t + i] != -1:
                            alignments[l][t] = alignments[l][t + i]
                        elif t - i >= 0 and alignments[l][t - i] != -1:
                            alignments[l][t] = alignments[l][t - i]
                        i += 1
                elif unaligned_target == "eps":
                    alignments[l][t] = eps_index
                elif unaligned_target == "keep":
                    pass

            # Process multiply aligned target
            if isinstance(alignments[l][t], list):
                if multiply_aligned_target == "bbn":
                    center = (len(alignments[l][t])) // 2
                    alignments[l][t] = alignments[l][t][center]
                elif multiply_aligned_target == "keep":
                    pass

    return alignments


def print_alignments(alignments,
                     stream,
                     eps_index,
                     print_unaligned_target: bool = False,
                     flat: bool = True):
    """
    print alignments to stream
    :param alignments: the alignments to print
    :param stream: the stream to print to
    :param eps_index: the eps_index used for unaligned target
    :param print_unaligned_target: print unaligned target (ignored in case of eps_index not None)
    :param flat: if true flat format is printed else multiline format is used
    :return: None
    """
    for line, alignment in enumerate(alignments):
        alignment_str = ""
        if not flat:
            alignment_str += "SENT: %d \n" % line
        for target in range(len(alignment)):
            if isinstance(alignment[target], list):
                for source in alignment[target]:
                    if print_unaligned_target \
                            or source != -1 \
                            or (eps_index is not None and source == eps_index):
                        alignment_str += "S %d %d " % (source, target)
                        alignment_str += "\n" if not flat else ""
            else:
                if print_unaligned_target \
                        or alignment[target] != -1 \
                        or (eps_index is not None and alignment[target] == eps_index):
                    alignment_str += "S %d %d " % (alignment[target], target)
                    alignment_str += "\n" if not flat else ""
        alignment_str += "\n"
        stream.write(alignment_str)
    stream.flush()


def add_parameters(params):
    data_params = params.add_argument_group("Data & I/O")
    data_params.add_argument('--source', '-s',
                             required=True,
                             type=regular_file(),
                             help='Source side of parallel training data.')
    data_params.add_argument('--target', '-t',
                             required=True,
                             type=regular_file(),
                             help='Target side of parallel training data.')
    data_params.add_argument('--alignment', '-a',
                             required=True,
                             type=regular_file(),
                             help='Flat alignment training file, alignment point e.g. S 0 1.')
    data_params.add_argument('--output-format', '-of',
                             default=None,
                             choices=list(OUTPUT_FORMAT),
                             help='Output format. Either one-line or multi-line. If not specified the same as input.'
                                  'Default: %(default)s.')
    data_params.add_argument('--output', '-o',
                             default=None,
                             type=str,
                             help='Output file to write alignments to. '
                                  'If not given, will write to stdout.')

    options_params = params.add_argument_group("Options")
    options_params.add_argument('--unaligned-target', '-ua',
                                default="keep",
                                choices=list(UNALIGNED_TARGET),
                                help="Mode to handle unaligned target words. "
                                     "Default: %(default)s.")
    options_params.add_argument('--multiply-aligned-target', '-ma',
                                default="keep",
                                choices=list(MULTIPLY_ALIGNED_TARGET),
                                help="Mode to handle multiply aligned target words. "
                                     "Default: %(default)s.")
    options_params.add_argument('--unaligned-target-epsilon-index', '-target-eps-index',
                                default=None,
                                type=int,
                                help="Index of epsilon. "
                                     "Default: %(default)s.")


def main():
    params = argparse.ArgumentParser(description='Alignment CLI')
    add_parameters(params)
    args = params.parse_args()

    trg_lengths = [len(x) for x in read_sentences(args.target)]
    src_lengths = [len(x) for x in read_sentences(args.source)]

    alignments, is_multiline = read_alignment_file(
        path=args.alignment,
        trg_lengths=trg_lengths,
        src_lengths=src_lengths)

    alignments = process_alignments(alignments=alignments,
                                    unaligned_target=args.unaligned_target,
                                    multiply_aligned_target=args.multiply_aligned_target,
                                    eps_index=args.unaligned_target_epsilon_index)

    if args.output_format is None:
        flat_output = not is_multiline
    else:
        flat_output = True if args.output_format == "flat" else False

    output_stream = sys.stdout if args.output is None else smart_open(args.output, mode='w')
    print_alignments(alignments=alignments,
                     stream=output_stream,
                     print_unaligned_target=True if args.unaligned_target == "keep" else False,
                     eps_index=args.unaligned_target_epsilon_index,
                     flat=flat_output)


if __name__ == '__main__':
    main()