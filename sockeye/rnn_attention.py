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
Implementations of different attention mechanisms in sequence-to-sequence models.
"""
import logging
from typing import Callable, NamedTuple, Optional, Tuple

import mxnet as mx

import config
import constants as C
import coverage
import layers
import utils

logger = logging.getLogger(__name__)


class AttentionConfig(config.Config):
    """
    Attention configuration.

    :param type: Attention name.
    :param num_hidden: Number of hidden units for attention networks.
    :param input_previous_word: Feeds the previous target embedding into the attention mechanism.
    :param source_num_hidden: Number of hidden units of the source.
    :param query_num_hidden: Number of hidden units of the query.
    :param layer_normalization: Apply layer normalization to MLP attention.
    :param config_coverage: Optional coverage configuration.
    :param num_heads: Number of attention heads. Only used for Multi-head dot attention.
    :param alignment_bias: rate of using alignment bias in training (applied batch-wise)
    :param alignment_assisted: concat source context selected using alignment with attention-weighted source context
    :param alignment_interpolation: flag for interpolation between attention and alignment distributions
    """
    def __init__(self,
                 type: str,
                 num_hidden: int,
                 input_previous_word: bool,
                 source_num_hidden: int,
                 query_num_hidden: int,
                 layer_normalization: bool,
                 config_coverage: Optional[coverage.CoverageConfig] = None,
                 num_heads: Optional[int] = None,
                 alignment_bias: bool = False,
                 alignment_assisted: float = 0.0,
                 alignment_interpolation: bool = False,
                 dynamic_alignment_interpolation: bool = False,
                 uniform_unaligned_context : bool = False,
                 last_aligned_context: bool = False) -> None:
        super().__init__()
        self.type = type
        self.num_hidden = num_hidden
        self.input_previous_word = input_previous_word
        self.source_num_hidden = source_num_hidden
        self.query_num_hidden = query_num_hidden
        self.layer_normalization = layer_normalization
        self.config_coverage = config_coverage
        self.num_heads = num_heads
        self.alignment_bias = alignment_bias
        self.alignment_assisted = alignment_assisted
        self.alignment_interpolation = alignment_interpolation
        self.dynamic_alignment_interpolation = dynamic_alignment_interpolation
        self.uniform_unaligned_context = uniform_unaligned_context
        self.last_aligned_context = last_aligned_context


def get_attention(config: AttentionConfig, max_seq_len: int) -> 'Attention':
    """
    Returns an Attention instance based on attention_type.

    :param config: Attention configuration.
    :param max_seq_len: Maximum length of source sequences.
    :return: Instance of Attention.
    """
    if config.type == C.ATT_BILINEAR:
        if config.input_previous_word:
            logger.warning("bilinear attention does not support input_previous_word")
        return BilinearAttention(config.query_num_hidden)
    elif config.type == C.ATT_DOT:
        return DotAttention(config.input_previous_word, config.source_num_hidden, config.query_num_hidden,
                            config.num_hidden)
    elif config.type == C.ATT_MH_DOT:
        utils.check_condition(config.num_heads is not None, "%s requires setting num-heads." % C.ATT_MH_DOT)
        return MultiHeadDotAttention(config.input_previous_word,
                                     num_hidden=config.num_hidden,
                                     heads=config.num_heads)
    elif config.type == C.ATT_DOT_SCALED:
        return DotAttention(config.input_previous_word, config.source_num_hidden, config.query_num_hidden,
                            config.num_hidden, scale=config.num_hidden ** -0.5)
    elif config.type == C.ATT_FIXED:
        return EncoderLastStateAttention(config.input_previous_word)
    elif config.type == C.ATT_LOC:
        return LocationAttention(config.input_previous_word, max_seq_len)
    elif config.type == C.ATT_MLP:
        return MlpAttention(input_previous_word=config.input_previous_word,
                            attention_num_hidden=config.num_hidden,
                            layer_normalization=config.layer_normalization,
                            alignment_bias=config.alignment_bias,
                            alignment_assisted=config.alignment_assisted,
                            alignment_interpolation = config.alignment_interpolation,
                            dynamic_alignment_interpolation=config.dynamic_alignment_interpolation)
    elif config.type == C.ATT_ALIGNMENT:
        return Alignment(input_previous_word=config.input_previous_word,
                         uniform_unaligned_context=config.uniform_unaligned_context,
                         last_aligned_context=config.last_aligned_context)
    elif config.type == C.ATT_COV:
        return MlpAttention(input_previous_word=config.input_previous_word,
                            attention_num_hidden=config.num_hidden,
                            layer_normalization=config.layer_normalization,
                            config_coverage=config.config_coverage,
                            alignment_bias=config.alignment_bias,
                            alignment_assisted=config.alignment_assisted,
                            alignment_interpolation=config.alignment_interpolation,
                            dynamic_alignment_interpolation=config.dynamic_alignment_interpolation)
    else:
        raise ValueError("Unknown attention type %s" % config.type)


AttentionInput = NamedTuple('AttentionInput', [('seq_idx', int), ('query', mx.sym.Symbol)])
"""
Input to attention callables.

:param seq_idx: Decoder time step / sequence index.
:param query: Query input to attention mechanism, e.g. decoder hidden state (plus previous word).
"""

AttentionState = NamedTuple('AttentionState', [
    ('context', mx.sym.Symbol),
    ('probs', mx.sym.Symbol),
    ('dynamic_source', mx.sym.Symbol),
])
"""
Results returned from attention callables.

:param context: Context vector (Bahdanau et al, 15). Shape: (batch_size, encoder_num_hidden)
:param probs: Attention distribution over source encoder states. Shape: (batch_size, source_seq_len).
:param dynamic_source: Dynamically updated source encoding.
       Shape: (batch_size, source_seq_len, dynamic_source_num_hidden)
"""


class Attention(object):
    """
    Generic attention interface that returns a callable for attending to source states.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param dynamic_source_num_hidden: Number of hidden units of dynamic source encoding update mechanism.
    """

    def __init__(self,
                 input_previous_word: bool,
                 dynamic_source_num_hidden: int = 1,
                 prefix: str = C.ATTENTION_PREFIX) -> None:
        self.dynamic_source_num_hidden = dynamic_source_num_hidden
        self._input_previous_word = input_previous_word
        self.prefix = prefix

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        def attend(att_input: AttentionInput, att_state: AttentionState,alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: source positions
            :param last_alignment: last aligned source positions
            :return: Updated attention state.
            """
            raise NotImplementedError()

        return attend

    def get_initial_state(self, source_length: mx.sym.Symbol, source_seq_len: int) -> AttentionState:
        """
        Returns initial attention state. Dynamic source encoding is initialized with zeros.

        :param source_length: Source length. Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        """
        dynamic_source = mx.sym.expand_dims(mx.sym.expand_dims(mx.sym.zeros_like(source_length), axis=1), axis=2)
        # dynamic_source: (batch_size, source_seq_len, num_hidden_dynamic_source)
        dynamic_source = mx.sym.broadcast_to(dynamic_source, shape=(0, source_seq_len, self.dynamic_source_num_hidden))
        return AttentionState(context=None, probs=None, dynamic_source=dynamic_source)

    def make_input(self,
                   seq_idx: int,
                   word_vec_prev: mx.sym.Symbol,
                   decoder_state: mx.sym.Symbol) -> AttentionInput:
        """
        Returns AttentionInput to be fed into the attend callable returned by the on() method.

        :param seq_idx: Decoder time step.
        :param word_vec_prev: Embedding of previously predicted ord
        :param decoder_state: Current decoder state
        :return: Attention input.
        """
        query = decoder_state
        if self._input_previous_word:
            # (batch_size, num_target_embed + rnn_num_hidden)
            query = mx.sym.concat(word_vec_prev, decoder_state, dim=1,
                                  name='%sconcat_prev_word_%d' % (self.prefix, seq_idx))
        return AttentionInput(seq_idx=seq_idx, query=query)


class BilinearAttention(Attention):
    """
    Bilinear attention based on Luong et al. 2015.

    :math:`score(h_t, h_s) = h_t^T \\mathbf{W} h_s`

    For implementation reasons we modify to:

    :math:`score(h_t, h_s) = h_s^T \\mathbf{W} h_t`

    :param num_hidden: Number of hidden units the source will be projected to.
    """

    def __init__(self, num_hidden: int) -> None:
        super().__init__(False)
        self.num_hidden = num_hidden
        self.s2t_weight = mx.sym.Variable("%ss2t_weight" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        # (batch_size, seq_len, self.num_hidden)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.s2t_weight,
                                              num_hidden=self.num_hidden,
                                              no_bias=True,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: aligned source positions
            :param last_alignment: last aligned source positions
            :return: Updated attention state.
            """
            # (batch_size, decoder_num_hidden, 1)
            query = mx.sym.expand_dims(att_input.query, axis=2)

            # in:  (batch_size, source_seq_len, self.num_hidden) X (batch_size, self.num_hidden, 1)
            # out: (batch_size, source_seq_len, 1).
            attention_scores = mx.sym.batch_dot(lhs=source_hidden, rhs=query, name="%sbatch_dot" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class DotAttention(Attention):
    """
    Attention mechanism with dot product between encoder and decoder hidden states [Luong et al. 2015].

    :math:`score(h_t, h_s) =  \\langle h_t, h_s \\rangle`

    :math:`a = softmax(score(*, h_s))`

    If rnn_num_hidden != num_hidden, states are projected with additional parameters to num_hidden.

    :math:`score(h_t, h_s) = \\langle \\mathbf{W}_t h_t, \\mathbf{W}_s h_s \\rangle`

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param source_num_hidden: Number of hidden units in source.
    :param query_num_hidden: Number of hidden units in query.
    :param num_hidden: Number of hidden units.
    :param scale: Optionally scale query before dot product [Vaswani et al, 2017].
    """

    def __init__(self,
                 input_previous_word: bool,
                 source_num_hidden: int,
                 query_num_hidden: int,
                 num_hidden: int,
                 scale: Optional[float] = None) -> None:
        super().__init__(input_previous_word)
        self.project_source = source_num_hidden != num_hidden
        self.project_query = query_num_hidden != num_hidden
        self.num_hidden = num_hidden
        self.scale = scale

        self.s2h_weight = mx.sym.Variable("%ss2h_weight" % self.prefix) if self.project_source else None
        self.t2h_weight = mx.sym.Variable("%st2h_weight" % self.prefix) if self.project_query else None

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        if self.project_source:
            # (batch_size, seq_len, self.num_hidden)
            source_hidden = mx.sym.FullyConnected(data=source,
                                                  weight=self.s2h_weight,
                                                  num_hidden=self.num_hidden,
                                                  no_bias=True,
                                                  flatten=False,
                                                  name="%ssource_hidden_fc" % self.prefix)
        else:
            source_hidden = source

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: aligned source positions
            :param last_alignment: last aligned source positions
            :return: Updated attention state.
            """
            query = att_input.query
            if self.project_query:
                # query: (batch_size, self.num_hidden)
                query = mx.sym.FullyConnected(data=query,
                                              weight=self.t2h_weight,
                                              num_hidden=self.num_hidden,
                                              no_bias=True, name="%squery_hidden_fc" % self.prefix)

            # scale down dot product by sqrt(num_hidden) [Vaswani et al, 17]
            if self.scale is not None:
                query = query * self.scale

            # (batch_size, decoder_num_hidden, 1)
            expanded_decoder_state = mx.sym.expand_dims(query, axis=2)

            # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.batch_dot(lhs=source_hidden, rhs=expanded_decoder_state,
                                                name="%sbatch_dot" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class MultiHeadDotAttention(Attention):
    """
    Dot product attention with multiple heads as proposed in Vaswani et al, Attention is all you need.
    Can be used with a RecurrentDecoder.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param num_hidden: Number of hidden units.
    :param heads: Number of attention heads / independently computed attention scores.
    """

    def __init__(self,
                 input_previous_word: bool,
                 num_hidden: int,
                 heads: int) -> None:
        super().__init__(input_previous_word)
        utils.check_condition(num_hidden % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, num_hidden))
        self.num_hidden = num_hidden
        self.heads = heads
        self.num_hidden_per_head = self.num_hidden // self.heads
        self.s2h_weight = mx.sym.Variable("%ss2h_weight" % self.prefix)
        self.s2h_bias = mx.sym.Variable("%ss2h_bias" % self.prefix)
        self.t2h_weight = mx.sym.Variable("%st2h_weight" % self.prefix)
        self.t2h_bias = mx.sym.Variable("%st2h_bias" % self.prefix)
        self.h2o_weight = mx.sym.Variable("%sh2o_weight" % self.prefix)
        self.h2o_bias = mx.sym.Variable("%sh2o_bias" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """
        # (batch, length, num_hidden * 2)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.s2h_weight,
                                              bias=self.s2h_bias,
                                              num_hidden=self.num_hidden * 2,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)
        # split keys and values
        # (batch, length, num_hidden)
        # pylint: disable=unbalanced-tuple-unpacking
        keys, values = mx.sym.split(data=source_hidden, num_outputs=2, axis=2)

        # (batch*heads, length, num_hidden/head)
        keys = layers.split_heads(keys, self.num_hidden_per_head, self.heads)
        values = layers.split_heads(values, self.num_hidden_per_head, self.heads)

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: aligned source positions
            :param last_alignment: last aligned source positions
            :return: Updated attention state.
            """
            # (batch, num_hidden)
            query = mx.sym.FullyConnected(data=att_input.query,
                                          weight=self.t2h_weight, bias=self.t2h_bias,
                                          num_hidden=self.num_hidden, name="%squery_hidden_fc" % self.prefix)
            # (batch, length, heads, num_hidden/head)
            query = mx.sym.reshape(query, shape=(0, 1, self.heads, self.num_hidden_per_head))
            # (batch, heads, num_hidden/head, length)
            query = mx.sym.transpose(query, axes=(0, 2, 3, 1))
            # (batch * heads, num_hidden/head, 1)
            query = mx.sym.reshape(query, shape=(-3, self.num_hidden_per_head, 1))

            # scale dot product
            query = query * (self.num_hidden_per_head ** -0.5)

            # (batch*heads, length, num_hidden/head) X (batch*heads, num_hidden/head, 1)
            #   -> (batch*heads, length, 1)
            attention_scores = mx.sym.batch_dot(lhs=keys, rhs=query, name="%sdot" % self.prefix)

            # (batch*heads, 1)
            lengths = layers.broadcast_to_heads(source_length, self.heads, ndim=1, fold_heads=True)

            # context: (batch*heads, num_hidden/head)
            # attention_probs: (batch*heads, length)
            context, attention_probs = get_context_and_attention_probs(values, lengths, attention_scores)

            # combine heads
            # (batch*heads, 1, num_hidden/head)
            context = mx.sym.expand_dims(context, axis=1)
            # (batch, 1, num_hidden)
            context = layers.combine_heads(context, self.num_hidden_per_head, heads=self.heads)
            # (batch, num_hidden)
            context = mx.sym.reshape(context, shape=(-3, -1))

            # (batch, heads, length)
            attention_probs = mx.sym.reshape(data=attention_probs, shape=(-4, -1, self.heads, source_seq_len))
            # just average over distributions
            attention_probs = mx.sym.mean(attention_probs, axis=1, keepdims=False)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class EncoderLastStateAttention(Attention):
    """
    Always returns the last encoder state independent of the query vector.
    Equivalent to no attention.
    """

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """
        source = mx.sym.swapaxes(source, dim1=0, dim2=1)
        encoder_last_state = mx.sym.SequenceLast(data=source, sequence_length=source_length,
                                                 use_sequence_length=True)
        fixed_probs = mx.sym.one_hot(source_length - 1, depth=source_seq_len)

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            return AttentionState(context=encoder_last_state,
                                  probs=fixed_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class LocationAttention(Attention):
    """
    Attends to locations in the source [Luong et al, 2015]

    :math:`a_t = softmax(\\mathbf{W}_a h_t)` for decoder hidden state at time t.

    :note: :math:`\\mathbf{W}_a` is of shape (max_source_seq_len, decoder_num_hidden).

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param max_source_seq_len: Maximum length of source sequences.
    """

    def __init__(self,
                 input_previous_word: bool,
                 max_source_seq_len: int) -> None:
        super().__init__(input_previous_word)
        self.max_source_seq_len = max_source_seq_len
        self.location_weight = mx.sym.Variable("%sloc_weight" % self.prefix)
        self.location_bias = mx.sym.Variable("%sloc_bias" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: aligned source positions
            :param last_alignment: last aligned source positions
            :return: Updated attention state.
            """
            # attention_scores: (batch_size, seq_len)
            attention_scores = mx.sym.FullyConnected(data=att_input.query,
                                                     num_hidden=self.max_source_seq_len,
                                                     weight=self.location_weight,
                                                     bias=self.location_bias)

            # attention_scores: (batch_size, seq_len)
            attention_scores = mx.sym.slice_axis(data=attention_scores,
                                                 axis=1,
                                                 begin=0,
                                                 end=source_seq_len)

            # attention_scores: (batch_size, seq_len, 1)
            attention_scores = mx.sym.expand_dims(data=attention_scores, axis=2)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class MlpAttention(Attention):
    """
    Attention computed through a one-layer MLP with num_hidden units [Luong et al, 2015].

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s] + b)`

    :math:`a = softmax(score(*, h_s))`

    Optionally, if attention_coverage_type is not None, attention uses dynamic source encoding ('coverage' mechanism)
    as in Tu et al. (2016): Modeling Coverage for Neural Machine Translation.

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s, c_s] + b)`

    :math:`c_s` is the decoder time-step dependent source encoding which is updated using the current
    decoder state.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param attention_num_hidden: Number of hidden units.
    :param layer_normalization: If true, normalizes hidden layer outputs before tanh activation.
    :param config_coverage: Optional coverage config.
    :param alignment_bias: Optional use of extdernal hard alignment bias to compuate attention weights. Value
                            determines how often this is used during training. When applied, it applies to
                            the whole batch.
    :param alignment_assisted: Optional boolean value to determine whether to concatenate the source context selected
                                using external hard alignment with the attention-weighted source context
    :param alignment_interpolation: Optional boolean value to set interpolation between alignment and attention distributions
    """

    def __init__(self,
                 input_previous_word: bool,
                 attention_num_hidden: int,
                 layer_normalization: bool = False,
                 config_coverage: Optional[coverage.CoverageConfig] = None,
                 alignment_bias: float = 0.0,
                 alignment_assisted: float = 0.0,
                 alignment_interpolation: bool = False,
                 dynamic_alignment_interpolation: bool = False) -> None:
        dynamic_source_num_hidden = 1 if config_coverage is None else config_coverage.num_hidden
        super().__init__(input_previous_word=input_previous_word,
                         dynamic_source_num_hidden=dynamic_source_num_hidden)
        self.attention_num_hidden = attention_num_hidden
        # input (encoder) to hidden
        self.att_e2h_weight = mx.sym.Variable("%se2h_weight" % self.prefix)
        # input (query) to hidden
        self.att_q2h_weight = mx.sym.Variable("%sq2h_weight" % self.prefix)
        # hidden to score
        self.att_h2s_weight = mx.sym.Variable("%sh2s_weight" % self.prefix)
        # coverage
        self.coverage = coverage.get_coverage(config_coverage) if config_coverage is not None else None
        #alignment bias
        self.alignment_bias = alignment_bias
        self.att_align_bias = mx.sym.Variable("%salign_bias" % self.prefix, shape=(attention_num_hidden,)) if alignment_bias else None
        self.att_align_interp_weight = mx.sym.Variable("%satt_align_interp_weight" % self.prefix, shape=(1,2),init=mx.init.Constant(0.5)) \
                                                    if alignment_interpolation else None
        self.att_align_dynamic_interp_weight = mx.sym.Variable("%satt_align_dynamic_interp_weight" % self.prefix, shape=( attention_num_hidden,1)) \
                                                    if dynamic_alignment_interpolation else None
        self.att_dynamic_interp_weight = mx.sym.Variable("%satt_dynamic_interp_weight" % self.prefix,
                                                               shape=(1, attention_num_hidden)) \
                                                    if dynamic_alignment_interpolation else None

        self.align_dynamic_interp_weight = mx.sym.Variable("%salign_dynamic_interp_weight" % self.prefix,
                                                               shape=(1,attention_num_hidden),
                                                               init =
                                                               mx.init.Constant(0)) \
                                                    if dynamic_alignment_interpolation else None
        # dynamic source (coverage) weights and settings
        # input (coverage) to hidden
        self.att_c2h_weight = mx.sym.Variable("%sc2h_weight" % self.prefix) if config_coverage is not None else None
        # layer normalization
        self._ln = layers.LayerNormalization(num_hidden=attention_num_hidden,
                                             prefix="%snorm" % self.prefix) if layer_normalization else None
        self.alignment_assisted = alignment_assisted
        self.alignment_layer = Alignment(input_previous_word,False) if (isinstance(alignment_assisted, float)
                                                                        and alignment_assisted > 0.0 )\
                                                                    or (isinstance(alignment_assisted, list)
                                                                           and max(alignment_assisted) > 0.0) else None
        self.alignment_interpolation = alignment_interpolation
        self.dynamic_alignment_interpolation = dynamic_alignment_interpolation
        
        self.alignment_func = None

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        coverage_func = self.coverage.on(source, source_length, source_seq_len) if self.coverage else None

        # (batch_size, seq_len, attention_num_hidden)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.att_e2h_weight,
                                              num_hidden=self.attention_num_hidden,
                                              no_bias=True,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)

        self.align_bias_prob = mx.sym.Custom(op_type="AlignBiasProb", low=0, high=1)

        if self.alignment_layer:
            self.align_assisted_prob = mx.sym.Custom(op_type="AlignBiasProb", low=0, high=1)
            self.alignment_func = self.alignment_layer.on(source,source_length,source_seq_len)

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: Shape: (batch_size,1)
            :param last_alignment: last aligned positions. Shape: (batch_size,1)
            :return: Updated attention state.
            """

            # (batch_size, attention_num_hidden)
            query_hidden = mx.sym.FullyConnected(data=att_input.query,
                                                 weight=self.att_q2h_weight,
                                                 num_hidden=self.attention_num_hidden,
                                                 no_bias=True,
                                                 name="%squery_hidden" % self.prefix)

            # (batch_size, 1, attention_num_hidden)
            query_hidden = mx.sym.expand_dims(data=query_hidden,
                                              axis=1,
                                              name="%squery_hidden_expanded" % self.prefix)

            attention_hidden_lhs = source_hidden
            if self.coverage:
                # (batch_size, seq_len, attention_num_hidden)
                dynamic_hidden = mx.sym.FullyConnected(data=att_state.dynamic_source,
                                                       weight=self.att_c2h_weight,
                                                       num_hidden=self.attention_num_hidden,
                                                       no_bias=True,
                                                       flatten=False,
                                                       name="%sdynamic_source_hidden_fc" % self.prefix)

                # (batch_size, seq_len, attention_num_hidden
                attention_hidden_lhs = dynamic_hidden + source_hidden

            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.broadcast_add(lhs=attention_hidden_lhs, rhs=query_hidden,
                                                    name="%squery_plus_input" % self.prefix)
            #self.debug_attention_hidden_before_bias[self.debug_cnt] = attention_hidden
            if self.alignment_bias > 0.0:
                #(batch_size, 1, seq_len)
                alignment_one_hot = mx.sym.one_hot(alignment,source_seq_len,name="%salignment_one_hot" % self.prefix)
                #(batch_size,attention_num_hidden,seq_len)
                alignment_one_hot = mx.sym.broadcast_to(data=alignment_one_hot,shape=(0,self.attention_num_hidden,0))
                #(batch_size,seq_len,attention_num_hidden)
                alignment_one_hot = mx.sym.swapaxes(alignment_one_hot,1,2)
                #(batch_size,seq_len,attention_num_hidden)
                seq_align_bias = mx.sym.broadcast_mul(rhs=self.att_align_bias, lhs=alignment_one_hot)
                #self.debug_alignment_one_hot[self.debug_cnt] = seq_align_bias
                #self.debug_cnt +=1
                condition = mx.sym.broadcast_add(self.alignment_bias > self.align_bias_prob,
                                                 mx.sym.zeros_like(attention_hidden))
                attention_hidden = mx.sym.where(condition, seq_align_bias + attention_hidden, attention_hidden)
                #self.debug_attention_hidden_after_bias[self.debug_cnt] = attention_hidden

            if self._ln is not None:
                attention_hidden = self._ln.normalize(attention_hidden)

            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%shidden" % self.prefix)

            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.FullyConnected(data=attention_hidden,
                                                     weight=self.att_h2s_weight,
                                                     num_hidden=1,
                                                     no_bias=True,
                                                     flatten=False,
                                                     name="%sraw_att_score_fc" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)

            dynamic_source = att_state.dynamic_source
            if self.coverage:
                # update dynamic source encoding
                # Note: this is a slight change to the Tu et al, 2016 paper: input to the coverage update
                # is the attention input query, not the previous decoder state.
                dynamic_source = coverage_func(prev_hidden=att_input.query,
                                               attention_prob_scores=attention_probs,
                                               prev_coverage=att_state.dynamic_source)

            if self.alignment_layer:
                align_context = self.alignment_func(att_input, att_state, alignment).context
                extended_context = mx.sym.where(self.alignment_assisted > self.align_assisted_prob,
                                                align_context,
                                                mx.sym.zeros_like(align_context))
                context = mx.sym.concat(context, extended_context)
            elif self.alignment_interpolation:
                context,attention_probs = self.get_context_and_align_interp_attention_probs(source, source_length,
                                                                                            attention_scores,
                                                                                            alignment,
                                                                                            source_seq_len)
            elif self.dynamic_alignment_interpolation:
                context, attention_probs = self.get_context_and_align_dynamic_interp_attention_probs(source, source_length,
                                                                                             attention_scores,
                                                                                             alignment,
                                                                                             source_seq_len)


            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=dynamic_source)

        return attend
    
#    def get_context_and_align_dynamic_interp_attention_probs(self, values: mx.sym.Symbol,
#                                                                 length: mx.sym.Symbol,
#                                                                 logits: mx.sym.Symbol,
#                                                                 alignment: mx.sym.Symbol,
#                                                                 source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
#
#
#        """
#        Returns context vector and attention probabilities after interpolation with alignment distribution
#        Softmax( lambda * attention_weights + (1-lambda) * alignment_one_hot)
#        where lambda = tanh (  alignment +  attention_weights))
#
#        :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
#        :param length: Shape: (batch_size,).
#        :param logits: Shape: (batch_size, seq_len, 1).
#        :param alignment: Shape: (batch_size,)
#        :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
#        """
#        # (batch_size, seq_len, 1)
#        logits = mask_attention_scores(logits, length)
#
#        # (batch_size, seq_len, 1)
#        probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')
#
#        # (batch_size, 1, seq_len)
#        alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)
#
#        # (batch_size,seq_len, 1)
#        alignment_one_hot = mx.sym.swapaxes(alignment_one_hot, 1, 2)
#
#        # (batch_size*seq_len,1)
#        probs = mx.sym.reshape(data=probs, shape=(-3, 0))
#        
#        # (batch_size*seq_len,1)
#        alignment_one_hot = mx.sym.reshape(data=alignment_one_hot, shape=(-3, 0))
#
#        # (batch_size*seq_len,1)
#        interp_weights = mx.sym.tanh(probs + alignment_one_hot)
#
#        interp_weights_complementary= 1 - interp_weights
#
#        interp_scores = interp_weights * probs + interp_weights_complementary * alignment_one_hot
#
#        # (batch_size,seq_len,1)
#        interp_scores = mx.sym.reshape_like(lhs=interp_scores, rhs=logits)
#
#        interp_scores = mask_attention_scores(interp_scores, length)
#        #(batch_size, seq_len, 1)
#        probs = mx.sym.softmax(interp_scores, axis=1, name='attention_align_interp_softmax')
#
#
#        # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
#        # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
#        context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
#        # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
#        context = mx.sym.reshape(data=context, shape=(0, 0))
#        probs = mx.sym.reshape(data=probs, shape=(0, 0))
#
#        return context, probs
    
  #  def get_context_and_align_dynamic_interp_attention_probs(self, values: mx.sym.Symbol,
  #                                                               length: mx.sym.Symbol,
  #                                                               logits: mx.sym.Symbol,
  #                                                               alignment: mx.sym.Symbol,
  #                                                               source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:


  #      """
  #      Returns context vector and attention probabilities after interpolation with alignment distribution
  #      Softmax( lambda * attention_weights + (1-lambda) * alignment_one_hot)
  #      where lambda = sigmoid (  alignment +  attention_weights))

  #      :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
  #      :param length: Shape: (batch_size,).
  #      :param logits: Shape: (batch_size, seq_len, 1).
  #      :param alignment: Shape: (batch_size,)
  #      :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
  #      """
  #      # (batch_size, seq_len, 1)
  #      logits = mask_attention_scores(logits, length)

  #      # (batch_size, seq_len, 1)
  #      probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

  #      # (batch_size, 1, seq_len)
  #      alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)

  #      # (batch_size,seq_len, 1)
  #      alignment_one_hot = mx.sym.swapaxes(alignment_one_hot, 1, 2)

  #      # (batch_size*seq_len,1)
  #      probs = mx.sym.reshape(data=probs, shape=(-3, 0))
  #      
  #      # (batch_size*seq_len,1)
  #      alignment_one_hot = mx.sym.reshape(data=alignment_one_hot, shape=(-3, 0))

  #      # (batch_size*seq_len,1)
  #      interp_weights = mx.sym.sigmoid(probs + alignment_one_hot)

  #      interp_weights_complementary= 1 - interp_weights

  #      interp_scores = interp_weights * probs + interp_weights_complementary * alignment_one_hot

  #      # (batch_size,seq_len,1)
  #      interp_scores = mx.sym.reshape_like(lhs=interp_scores, rhs=logits)

  #      interp_scores = mask_attention_scores(interp_scores, length)
  #      #(batch_size, seq_len, 1)
  #      probs = mx.sym.softmax(interp_scores, axis=1, name='attention_align_interp_softmax')


  #      # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
  #      # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
  #      context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
  #      # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
  #      context = mx.sym.reshape(data=context, shape=(0, 0))
  #      probs = mx.sym.reshape(data=probs, shape=(0, 0))

  #      return context, probs

    def get_context_and_align_dynamic_interp_attention_probs(self, values: mx.sym.Symbol,
                                                                 length: mx.sym.Symbol,
                                                                 logits: mx.sym.Symbol,
                                                                 alignment: mx.sym.Symbol,
                                                                 source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:


        """
        Returns context vector and attention probabilities after interpolation with alignment distribution
        Softmax( lambda * attention_weights + (1-lambda) * alignment_one_hot)
        where lambda = sigmoid ( V^T tanh ( W alignment + M attention_weights))

        :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param length: Shape: (batch_size,).
        :param logits: Shape: (batch_size, seq_len, 1).
        :param alignment: Shape: (batch_size,)
        :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
        """
        # (batch_size, seq_len, 1)
        logits = mask_attention_scores(logits, length)

        # (batch_size, seq_len, 1)
        probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

        # (batch_size, 1, seq_len)
        alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)

        # (batch_size,seq_len, 1)
        alignment_one_hot = mx.sym.swapaxes(alignment_one_hot, 1, 2)

        # (batch_size*seq_len,1)
        probs = mx.sym.reshape(data=probs, shape=(-3, 0))
        # (batch_size*seq_len,num_hidden)
        interp_lhs_raw = mx.sym.linalg_gemm2(probs, self.att_dynamic_interp_weight)

        # (batch_size*seq_len,1)
        alignment_one_hot = mx.sym.reshape(data=alignment_one_hot, shape=(-3, 0))
        # (batch_size*seq_len,num_hidden)
        interp_rhs_raw = mx.sym.linalg_gemm2(alignment_one_hot, self.align_dynamic_interp_weight)

        interp_weights = mx.sym.tanh(interp_lhs_raw + interp_rhs_raw)

        # (batch_size*seq_len,1)
        interp_weights = mx.sym.linalg_gemm2(interp_weights,self.att_align_dynamic_interp_weight)
        interp_weights = mx.sym.sigmoid(interp_weights)

        interp_weights_complementary= 1 - interp_weights

        interp_scores = interp_weights * probs + interp_weights_complementary * alignment_one_hot

        # (batch_size,seq_len,1)
        interp_scores = mx.sym.reshape_like(lhs=interp_scores, rhs=logits)

        interp_scores = mask_attention_scores(interp_scores, length)
        #(batch_size, seq_len, 1)
        probs = mx.sym.softmax(interp_scores, axis=1, name='attention_align_interp_softmax')


        # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
        # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
        context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
        # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
        context = mx.sym.reshape(data=context, shape=(0, 0))
        probs = mx.sym.reshape(data=probs, shape=(0, 0))

        return context, probs

    # def get_context_and_align_dynamic_interp_attention_probs(self, values: mx.sym.Symbol,
    #                                                      length: mx.sym.Symbol,
    #                                                      logits: mx.sym.Symbol,
    #                                                      alignment: mx.sym.Symbol,
    #                                                      source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
    #
    #
    #         """
    #         Returns context vector and attention probabilities after interpolation with alignment distribution
    #           interpolation weights are not normalized
    #
    #         :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
    #         :param length: Shape: (batch_size,).
    #         :param logits: Shape: (batch_size, seq_len, 1).
    #         :param alignment: Shape: (batch_size,)
    #         :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
    #         """
    #         # (batch_size, seq_len, 1)
    #         logits = mask_attention_scores(logits, length)
    #
    #         # (batch_size, seq_len, 1)
    #         probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')
    #
    #         # (batch_size, 1, seq_len)
    #         alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)
    #
    #         # (batch_size,seq_len, 1)
    #         alignment_one_hot = mx.sym.swapaxes(alignment_one_hot, 1, 2)
    #
    #         # (batch_size,seq_len,1)
    #         attention_align_sum = probs + alignment_one_hot
    #
    #         #(batch_size,seq_len,2)
    #         attention_align_concat = mx.sym.concat(probs,alignment_one_hot,dim=2)
    #         # (batch_size*seq_len,2)
    #         attention_align_concat = mx.sym.reshape(data=attention_align_concat, shape=(-3,0))
    #
    #         # (batch_size*seq_len,1)
    #         attention_align_sum = mx.sym.reshape(data=attention_align_sum, shape=(-3,0))
    #
    #         # (batch_size*seq_len,2)
    #         interp_raw_scores = mx.sym.linalg_gemm2(attention_align_sum,self.att_align_dynamic_interp_weight)
    #         interp_raw_scores = mx.sym.sigmoid(interp_raw_scores)
    #
    #         # (batch_size*seq_len,1,2)
    #         interp_raw_scores = mx.sym.expand_dims(data=interp_raw_scores, axis=1)
    #         # (batch_size*seq_len,2,1)
    #         attention_align_concat = mx.sym.expand_dims(data=attention_align_concat,axis=2)
    #
    #         # (batch_size*seq_len,1,1)
    #         interp_raw_scores = mx.sym.batch_dot(lhs=interp_raw_scores, rhs=attention_align_concat, transpose_a=False)
    #
    #         # (batch_size,seq_len,1)
    #         interp_raw_scores = mx.sym.reshape_like(lhs=interp_raw_scores,rhs=probs)
    #
    #         interp_raw_scores = mask_attention_scores(interp_raw_scores, length)
    #
    #         # (batch_size, seq_len, 1)
    #         probs = mx.sym.softmax(interp_raw_scores, axis=1, name='attention_align_interp_softmax')
    #
    #         # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
    #         # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
    #         context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
    #         # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
    #         context = mx.sym.reshape(data=context, shape=(0, 0))
    #         probs = mx.sym.reshape(data=probs, shape=(0, 0))
    #
    #         return context, probs
    
    def get_context_and_align_plus_attention_probs(self,  values: mx.sym.Symbol,
                                                            length: mx.sym.Symbol,
                                                            logits: mx.sym.Symbol,
                                                            alignment: mx.sym.Symbol,
                                                            source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:


            """
            Returns context vector and attention probabilities after interpolation with alignment distribution

            :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
            :param length: Shape: (batch_size,).
            :param logits: Shape: (batch_size, seq_len, 1).
            :param alignment: Shape: (batch_size,)
            :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
            """
            # (batch_size, seq_len, 1)
            logits = mask_attention_scores(logits, length)

            # (batch_size, seq_len, 1)
            probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

            #(batch_size, 1, seq_len)
            alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)

            # (batch_size,seq_len, 1)
            alignment_one_hot = mx.sym.swapaxes(alignment_one_hot,1,2)

            #(batch_size,seq_len,2)
            attention_align_sum = alignment_one_hot + probs

            attention_align_sum = mask_attention_scores(attention_align_sum, length)

            # (batch_size, seq_len, 1)
            probs = mx.sym.softmax(attention_align_sum, axis=1,
                    name='attention_plus_align_softmax')

            # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
            # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
            context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
            # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
            context = mx.sym.reshape(data=context, shape=(0, 0))
            probs = mx.sym.reshape(data=probs, shape=(0, 0))

            return context, probs

    def get_context_and_align_interp_attention_probs(self,  values: mx.sym.Symbol,
                                                            length: mx.sym.Symbol,
                                                            logits: mx.sym.Symbol,
                                                            alignment: mx.sym.Symbol,
                                                            source_seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:


            """
            Returns context vector and attention probabilities after interpolation with alignment distribution

            :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
            :param length: Shape: (batch_size,).
            :param logits: Shape: (batch_size, seq_len, 1).
            :param alignment: Shape: (batch_size,)
            :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
            """
            # (batch_size, seq_len, 1)
            logits = mask_attention_scores(logits, length)

            # (batch_size, seq_len, 1)
            probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

            #(batch_size, 1, seq_len)
            alignment_one_hot = mx.sym.one_hot(alignment, source_seq_len, name="%salignment_one_hot" % self.prefix)

            # (batch_size,seq_len, 1)
            alignment_one_hot = mx.sym.swapaxes(alignment_one_hot,1,2)

            #(batch_size,seq_len,2)
            attention_align_concat = mx.sym.concat(probs,alignment_one_hot,dim=2)

            #(batch_size,seq_len,1)
            interp_raw_scores = mx.sym.FullyConnected(data=attention_align_concat,
                                                     weight=self.att_align_interp_weight,
                                                     num_hidden=1,
                                                     no_bias=True,
                                                     flatten=False,
                                                     name="%sinterp_att_align_weight_fc" % self.prefix)

            interp_raw_scores = mask_attention_scores(interp_raw_scores, length)


            # (batch_size, seq_len, 1)
            probs = mx.sym.softmax(interp_raw_scores, axis=1, name='attention_align_interp_softmax')

            # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
            # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
            context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
            # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
            context = mx.sym.reshape(data=context, shape=(0, 0))
            probs = mx.sym.reshape(data=probs, shape=(0, 0))

            return context, probs


class Alignment(Attention):
    """
    Attention computed through a one-layer MLP with num_hidden units [Luong et al, 2015].

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s] + b)`

    :math:`a = softmax(score(*, h_s))`

    Optionally, if attention_coverage_type is not None, attention uses dynamic source encoding ('coverage' mechanism)
    as in Tu et al. (2016): Modeling Coverage for Neural Machine Translation.

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s, c_s] + b)`

    :math:`c_s` is the decoder time-step dependent source encoding which is updated using the current
    decoder state.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param attention_num_hidden: Number of hidden units.
    :param layer_normalization: If true, normalizes hidden layer outputs before tanh activation.
    :param config_coverage: Optional coverage config.
    """

    def __init__(self,input_previous_word: bool,
                 uniform_unaligned_context: bool,
                 last_aligned_context: bool = False) -> None:
        super().__init__(input_previous_word=input_previous_word)
        self.uniform_unaligned_context = uniform_unaligned_context
        self.last_aligned_context = last_aligned_context

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        def attend(att_input: AttentionInput, att_state: AttentionState,
                   alignment: mx.sym.Symbol = None,
                   last_alignment: mx.sym.Symbol = None) -> AttentionState:
            """
            Returns aligned context.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :param alignment: Shape: (batch_size,)
            :param last_alignment: last aligned positions. Shape (batch_size,)
            :return: Updated attention state.
            """

            one_hot  = mx.sym.one_hot(last_alignment if self.last_aligned_context else alignment,
                                      source_seq_len,
                                      name="%salignment_one_hot" % self.prefix)
            one_hot = mx.sym.swapaxes(one_hot, 1, 2)
            if self.uniform_unaligned_context:
                # (batch_size,)
                uniform_weights = mx.sym.ones_like(alignment)/mx.sym.expand_dims(data=source_length,axis=1)
                # (batch_size,1)
                uniform_weights = mx.sym.expand_dims(data=uniform_weights,axis=1)
                # (batch_size,source_seq_len)
                uniform_weights = mx.sym.broadcast_axis(data=uniform_weights,axis=1,size=source_seq_len)
                uniform_weights = mx.sym.swapaxes(uniform_weights, 0, 1)
                uniform_weights = mx.sym.SequenceMask(data=uniform_weights,
                                                      use_sequence_length=True,
                                                      sequence_length=source_length,
                                                      value=0)
                uniform_weights = mx.sym.swapaxes(uniform_weights, 0, 1)

                # determine unaligned positions
                # (batch_size,)
                unaligned = mx.sym.where(alignment >= 0, mx.sym.zeros_like(alignment), mx.sym.ones_like(alignment))
                # (batch_size,1)
                unaligned = mx.sym.expand_dims(data=unaligned,axis=1)
                # (batch_size,source_seq_len)
                unaligned = mx.sym.broadcast_axis(data=unaligned,axis=1,size=source_seq_len)

                # select between unifrom weights and one_hot selection
                one_hot = mx.sym.where(unaligned, uniform_weights, one_hot)


            context = mx.sym.batch_dot(lhs=source, rhs=one_hot, transpose_a=True)
            # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
            context = mx.sym.reshape(data=context, shape=(0, 0))
            attention_probs = mx.sym.reshape(data=one_hot, shape=(0, 0))

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


def mask_attention_scores(logits: mx.sym.Symbol,
                          length: mx.sym.Symbol) -> mx.sym.Symbol:
    """
    Masks attention scores according to sequence length.

    :param logits: Shape: (batch_size, seq_len, 1).
    :param length: Shape: (batch_size,).
    :return: Masked logits: (batch_size, seq_len, 1).
    """
    # TODO: Masking with 0-1 mask, to avoid the multiplication
    logits = mx.sym.swapaxes(data=logits, dim1=0, dim2=1)
    logits = mx.sym.SequenceMask(data=logits,
                                 use_sequence_length=True,
                                 sequence_length=length,
                                 value=C.LARGE_NEGATIVE_VALUE)
    # (batch_size, seq_len, 1)
    return mx.sym.swapaxes(data=logits, dim1=0, dim2=1)


def get_context_and_attention_probs(values: mx.sym.Symbol,
                                    length: mx.sym.Symbol,
                                    logits: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
    """
    Returns context vector and attention probabilities
    via a weighted sum over values.

    :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
    :param length: Shape: (batch_size,).
    :param logits: Shape: (batch_size, seq_len, 1).
    :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
    """
    # (batch_size, seq_len, 1)
    logits = mask_attention_scores(logits, length)

    # (batch_size, seq_len, 1)
    probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

    # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
    # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
    context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
    # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
    context = mx.sym.reshape(data=context, shape=(0, 0))
    probs = mx.sym.reshape(data=probs, shape=(0, 0))

    return context, probs
