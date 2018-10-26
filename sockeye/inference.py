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
Code for inference/translation
"""
import itertools
import copy
import logging
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set

import mxnet as mx
import numpy as np

import constants as C
import data_io
import lexicon
import model
import utils
import vocab
import decoder

logger = logging.getLogger(__name__)


def align_idx_offset(step):
    return max(step - 1 - C.MAX_JUMP, 0)


class InferenceModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param model_folder: Folder to load model from.
    :param context: MXNet context to bind modules to.
    :param beam_size: Beam size.
    :param batch_size: Batch size.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in model_folder.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    """

    def __init__(self,
                 model_folder: str,
                 context: mx.context.Context,
                 beam_size: int,
                 batch_size: int,
                 checkpoint: Optional[int] = None,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False,
                 vis_target_enc_attention_layer: int = 0) -> None:
        self.model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", self.model_version)
        utils.check_version(self.model_version)

        config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
        # if used at all, alignment bias should be on all the time in decoding
        #if config.config_decoder.attention_config.alignment_bias > 0.0:
        #                config.config_decoder.attention_config.alignment_bias = 1.0

        super().__init__(config)

        self.fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)

        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')

        self.beam_size = beam_size
        self.softmax_temperature = softmax_temperature
        self.batch_size = batch_size
        self.context = context
        self.alignment_based= self.config.config_data.alignment is not None
        self.use_unaligned = hasattr(self.config.config_decoder, "attention_config")\
                                and (self.config.config_decoder.attention_config.uniform_unaligned_context or \
                                self.config.config_decoder.attention_config.last_aligned_context)
        self.alignment_model = self.config.output_classes == C.ALIGNMENT_JUMP
        self._build_model_components()

        self.max_input_length, self.get_max_output_length = get_max_input_output_length([self],
                                                                                        max_output_length_num_stds)

        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_data_shapes_cache = None  # type: Optional[Dict]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: mx.nd.NDArray
        self.output_layer_b = None  # type: mx.nd.NDArray

        self.vis_target_enc_attention_layer = vis_target_enc_attention_layer

    def initialize(self, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_input_length = max_input_length
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()

        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.fname_params)
        self.encoder_module.init_params(arg_params=self.params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, allow_missing=False)

        if self.cache_output_layer_w_b:
            if self.output_layer.weight_normalization:
                # precompute normalized output layer weight imperatively
                assert self.output_layer.weight_norm is not None
                weight = self.params[self.output_layer.weight_norm.weight.name].as_in_context(self.context)
                scale = self.params[self.output_layer.weight_norm.scale.name].as_in_context(self.context)
                self.output_layer_w = self.output_layer.weight_norm(weight, scale)
            else:
                self.output_layer_w = self.params[self.output_layer.w.name].as_in_context(self.context)
            self.output_layer_b = self.params[self.output_layer.b.name].as_in_context(self.context)

    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_length = utils.compute_lengths(source)

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            # TODO(fhieber): Consider standardizing encoders to return batch-major data to avoid this line.
            source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len)

            data_names = [C.SOURCE_NAME]
            label_names = []  # type: List[str]
            return mx.sym.Group(decoder_init_states), data_names, label_names

        default_bucket_key = self.max_input_length + (1 if self.alignment_model else 0)
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key ,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int]]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the source sequence
        and the current time-step in the inference procedure (e.g. beam search).
        The latter corresponds to the current length of the target sequences.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int]):
            """
            Returns either softmax output (probs over target vocabulary) or inputs to logit
            computation, controlled by decoder_return_logit_inputs
            """
            source_seq_len, decode_step = bucket_key
            source_embed_seq_len = self.embedding_source.get_encoded_seq_len(source_seq_len)
            source_encoded_seq_len = self.encoder.get_encoded_seq_len(source_embed_seq_len)

            self.decoder.reset()
            target_prev = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables(decode_step)
            state_names = [state.name for state in states]

            # embedding for previous word
            # (batch_size, num_embed)
            target_embed_prev, _, _ = self.embedding_target.encode(data=target_prev, data_length=None, seq_len=1)

            #output embedding
            output_embed_prev = None
            if self.alignment_model and self.embedding_output is not None:
                label_prev = mx.sym.Variable(C.ALIGNMENT_JUMP_LABEL_NAME)
                (output_embed_prev, _, _ ) = self.embedding_output.encode(data=label_prev, data_length=None, seq_len=1)

            alignment = mx.sym.Variable(C.ALIGNMENT_NAME) if self.alignment_based else None
            last_alignment = mx.sym.Variable(C.LAST_ALIGNMENT_NAME) if self.alignment_based else None

            if hasattr(self.decoder, "vis_target_enc_attention_layer"):
                self.decoder.vis_target_enc_attention_layer = self.vis_target_enc_attention_layer

            # decoder
            # target_decoded: (batch_size, decoder_depth)
            (target_decoded,
             attention_probs,
             states) = self.decoder.decode_step(decode_step,
                                                target_embed_prev,
                                                source_encoded_seq_len,
                                                *states,
                                                alignment=alignment,
                                                last_alignment=last_alignment,
                                                output_embed_prev=output_embed_prev)

            if self.decoder_return_logit_inputs:
                # skip output layer in graph
                outputs = mx.sym.identity(target_decoded, name=C.LOGIT_INPUTS_NAME)
            else:
                # logits: (batch_size, target_vocab_size)
                logits = self.output_layer(target_decoded)
                if self.softmax_temperature is not None:
                    logits /= self.softmax_temperature
                outputs = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_NAME] + state_names +\
                            ([C.LAST_ALIGNMENT_NAME
                                if self.alignment_model
                                    and isinstance(self.config.config_decoder,decoder.RecurrentDecoderConfig)
                                    and self.config.config_decoder.attention_config.last_aligned_context==True
                                else C.ALIGNMENT_NAME]
                             if self.alignment_based else []) + \
                         ([C.ALIGNMENT_JUMP_LABEL_NAME] if self.alignment_model
                                                           and isinstance(self.config.config_decoder,
                                                                          decoder.RecurrentDecoderConfig)
                                                           and self.config.config_decoder.label_num_layers > 0
                          else [] )

            label_names = []  # type: List[str]
            return mx.sym.Group([outputs, attention_probs] + states), data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = (self.max_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_encoder_data_shapes(self, bucket_key: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(self.batch_size, bucket_key),
                               layout=C.BATCH_MAJOR)]

    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.
        Caches results for bucket_keys if called iteratively.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        return self.decoder_data_shapes_cache.setdefault(
            bucket_key,
            [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size * self.beam_size,), layout="NT")] +
            self.decoder.state_shapes(self.batch_size * self.beam_size,
                                      target_max_length,
                                      self.encoder.get_encoded_seq_len(source_max_length),
                                      self.encoder.get_num_hidden()) +
            ([mx.io.DataDesc(name=C.LAST_ALIGNMENT_NAME
                                    if self.alignment_model \
                                        and isinstance(self.config.config_decoder, decoder.RecurrentDecoderConfig)
                                        and self.config.config_decoder.attention_config.last_aligned_context==True
                                    else C.ALIGNMENT_NAME,
                            shape=(self.batch_size * self.beam_size,1),
                            layout="NT")] if self.alignment_based else []) +
            ([mx.io.DataDesc(name=C.ALIGNMENT_JUMP_LABEL_NAME,
                            shape=(self.batch_size * self.beam_size, 1),
                            layout="NT")] if self.alignment_model \
                                    and isinstance(self.config.config_decoder, decoder.RecurrentDecoderConfig)
                                    and self.config.config_decoder.label_num_layers > 0 else [])
        )


    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> 'ModelState':
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()
        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.repeat(s, repeats=self.beam_size, axis=0) for s in decoder_states]
        return ModelState(decoder_states)

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: Tuple[int, int],
                    model_state: 'ModelState',
                    step: int,
                    prev_alignment: mx.nd.NDArray = None,
                    last_alignment: mx.nd.NDArray = None,
                    previous_jump: mx.nd.NDArray = None,
                    actual_source_length: List[int] = [],
                    use_unaligned: bool = True,
                    skip_alignments: List[bool] = []) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState', mx.nd.NDArray]:
        """
        Runs forward pass of the single-step decoder.

        :return: Decoder stack output (logit inputs or probability distribution), attention scores, updated model state.
        """
        #logger.info("%s" % (locals()))
        #logger.info("offset %d" % align_idx_offset(step))
        #lexical alignment-based model: alignments hypothesized
        #alignment model: previous alignments used
        alignment_begin_idx = -1 if use_unaligned else 0
        alignment_max_length = 1
        new_bucket_key = (bucket_key[0] + (1 if self.alignment_model else 0),bucket_key[1])
        if self.alignment_based and not self.alignment_model:
            # TODO correct usage of max(actual_source_length)
            alignment_end_idx = max(0, min(C.MAX_JUMP, max(actual_source_length) - step) + min(C.MAX_JUMP, step - 1)) + 1
            if use_unaligned:
                alignment_end_idx += 1
            #print("range (%d, %d)"%(alignment_begin_idx, alignment_end_idx))
            #alignment_max_length = 2*C.MAX_JUMP + 2  #extra position for handling unaliged target words
            alignment_max_length = alignment_end_idx
        elif self.alignment_model:
            if use_unaligned:
                alignment_end_idx = 0
            else:
                alignment_end_idx = 1
        else:
            alignment_end_idx = 1



        #alignment_end_idx = max(actual_source_length) if self.alignment_based and not self.alignment_model else 0
        #extra position for handling unaliged target words
        #alignment_max_length = bucket_key[0]+1 if self.alignment_based and not self.alignment_model else 1
        alignment_shape = None
        for e in self._get_decoder_data_shapes(new_bucket_key):
            if e[0] == C.ALIGNMENT_NAME:
                alignment_shape = e[1]
                break

        out_result , attention_probs_result , alignment_result, model_state_result = None, None, None, None
        new_states = [None] * len(model_state.states)
        previous_jump = [previous_jump] if previous_jump is not None else []
        #if self.alignment_based and not self.alignment_model:
        #    print(step, "", alignment_end_idx)

        end = 0
        skipped = 0
        evaluated_positions = 0
        for align_pos in range(alignment_begin_idx,alignment_end_idx):
            align_idx = align_idx_offset(step) + align_pos if align_pos >= 0 else align_pos
            j = align_pos + 1 if use_unaligned else align_pos
            end = j
            #print(step, align_pos, j, align_idx)

            alignment = []
            if self.alignment_based:
                if self.alignment_model:
                    # shift by 1 due to BOS
                    if isinstance(self.config.config_decoder,decoder.RecurrentDecoderConfig) and \
                            self.config.config_decoder.attention_config.last_aligned_context:
                        alignment = [last_alignment + 1]
                    else:
                        alignment = [prev_alignment + 1]
                else:
                    #hypothesize alignment

                    if len(skip_alignments) > align_idx \
                            and skip_alignments[align_idx]\
                            and not np.all(skip_alignments[align_idx_offset(step):alignment_end_idx+align_idx_offset(step)])\
                            and not align_idx == actual_source_length: # always hypothesize sentence end
                        #logger.info("skip alignment point %d" % align_idx)
                        skipped += 1
                        continue

                    alignment = [align_idx*mx.ndarray.ones(ctx=self.context,shape=alignment_shape,dtype='int32')]
                    if alignment_result is None:
                        alignment_result = mx.ndarray.zeros(ctx=self.context, shape=(alignment_max_length, *(alignment[0].shape)), dtype='int32')
                    alignment_result[j,:,:] = alignment[0]
                    #alignment_result.append(copy.deepcopy(alignment))

            evaluated_positions += 1
            batch = mx.io.DataBatch(
                data=[prev_word.as_in_context(self.context)] + model_state.states + alignment + previous_jump,
                label=None,
                bucket_key=new_bucket_key,
                provide_data=self._get_decoder_data_shapes(new_bucket_key))
            self.decoder_module.forward(data_batch=batch, is_train=False)
            out, attention_probs, *new_states = self.decoder_module.get_outputs()
            if out_result is None:
                out_result = mx.ndarray.zeros(ctx=self.context, shape=(alignment_max_length, *(out.shape)),  dtype='float32')
            if attention_probs_result is None:
                attention_probs_result = mx.ndarray.zeros(ctx=self.context,shape=(alignment_max_length,*(attention_probs.shape)),dtype='float32')
            for sent in range(len(actual_source_length)):
                # if batch_size > 1 , additional positions might have been computed and need to be disregarded
                # check if actual alignment point is inside the length
                if align_idx < actual_source_length[sent]:
                    out_result[j,sent*self.beam_size:(sent+1)*self.beam_size,:] = out[sent*self.beam_size:(sent+1)*self.beam_size]
            attention_probs_result[j,:,:] = attention_probs
            #out_result.append(copy.deepcopy(out))
            #attention_probs_result.append(copy.deepcopy(attention_probs))
            if model_state_result is None:
                model_state_result = [mx.ndarray.zeros(ctx=self.context,shape=(alignment_max_length,*(state.shape)),dtype=state.dtype) for state in new_states]
            for state_idx in range(len(new_states)):
                model_state_result[state_idx][j] = new_states[state_idx]
            #model_state_result.append(copy.deepcopy(model_state))
            #model_state_result.append([ copy.deepcopy(e) for e in model_state.states])

        #print("end", end, out_result)
        # logger.info("skip %d out of %d alignment points. %d suggested" % (skipped, alignment_end_idx - alignment_begin_idx, mx.nd.sum(skip_alignments > 0)))
        if not self.alignment_model:
            logger.info("Evaluated positions = %d" % evaluated_positions)

        return out_result, attention_probs_result, ModelState(model_state_result), alignment_result

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        if self.config.config_data.max_observed_source_seq_len is not None:
            return self.config.config_data.max_observed_source_seq_len
        else:
            return self.config.max_seq_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        if self.config.config_data.max_observed_target_seq_len is not None:
            return self.config.config_data.max_observed_target_seq_len
        else:
            return self.config.max_seq_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.length_ratio_std


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                beam_size: int,
                batch_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False,
                vis_target_enc_attention_layer: int = 0) -> Tuple[List[InferenceModel], Dict[str, int], Dict[str, int]]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations to add to mean target-source length ratio
           to compute maximum output length.
    :param decoder_return_logit_inputs: Model decoders return inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Models cache weights and biases for logit computation as NumPy arrays (used with
                                   restrict lexicon).
    :return: List of models, source vocabulary, target vocabulary.
    """
    models, source_vocabs, target_vocabs = [], [], []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    for model_folder, checkpoint in zip(model_folders, checkpoints):
        source_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_SRC_NAME)))
        target_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_TRG_NAME)))
        model = InferenceModel(model_folder=model_folder,
                               context=context,
                               beam_size=beam_size,
                               batch_size=batch_size,
                               softmax_temperature=softmax_temperature,
                               checkpoint=checkpoint,
                               decoder_return_logit_inputs=decoder_return_logit_inputs,
                               cache_output_layer_w_b=cache_output_layer_w_b,
                               vis_target_enc_attention_layer=vis_target_enc_attention_layer)
        #batching disabled for alignment-based models for now
        #assert not model.alignment_based or batch_size ==1
        models.append(model)

    utils.check_condition(all(set(vocab.items()) == set(source_vocabs[0].items()) for vocab in source_vocabs),
                          "Source vocabulary ids do not match")
    utils.check_condition(all(set(vocab.items()) == set(target_vocabs[0].items()) for vocab in target_vocabs),
                          "Target vocabulary ids do not match")

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = get_max_input_output_length(models,
                                                                       max_output_length_num_stds,
                                                                       max_input_len)
    for model in models:
        model.initialize(max_input_len, get_max_output_length)

    return models, source_vocabs[0], target_vocabs[0]


def get_max_input_output_length(models: List[InferenceModel], num_stds: int,
                                max_input_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param max_input_len: An optional overwrite of the maximum input length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)

    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
    else:
        factor = max_mean + (max_std * num_stds)

    supported_max_seq_len_source = min((model.max_supported_seq_len_source for model in models
                                        if model.max_supported_seq_len_source is not None),
                                       default=None)
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)

    training_max_seq_len_source = min(model.training_max_seq_len_source for model in models)

    if max_input_len is None:
        # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
        # constraint. This is for example the case for learned positional embeddings, which are only defined for the
        # maximum source and target sequence length observed during training.
        if supported_max_seq_len_source is not None and supported_max_seq_len_target is None:
            max_input_len = supported_max_seq_len_source
        elif supported_max_seq_len_source is None and supported_max_seq_len_target is not None:
            if np.ceil(factor * training_max_seq_len_source) > supported_max_seq_len_target:
                max_input_len = int(np.floor(supported_max_seq_len_target / factor))
            else:
                max_input_len = training_max_seq_len_source
        elif supported_max_seq_len_source is not None or supported_max_seq_len_target is not None:
            if np.ceil(factor * supported_max_seq_len_source) > supported_max_seq_len_target:
                max_input_len = int(np.floor(supported_max_seq_len_target / factor))
            else:
                max_input_len = supported_max_seq_len_source
        else:
            # Any source/target length is supported and max_input_len was not manually set, therefore we use the
            # maximum length from training.
            max_input_len = training_max_seq_len_source

    def get_max_output_length(input_length: int):
        return int(np.ceil(factor * input_length))

    return max_input_len, get_max_output_length


Tokens = List[str]
TranslatorInput = NamedTuple('TranslatorInput', [
    ('id', int),
    ('sentence', str),
    ('tokens', Tokens),
    ('reference_tokens', Tokens),
])
"""
Required input for Translator.

:param id: Sentence id.
:param sentence: Input sentence.
:param tokens: List of input tokens.
"""

InputChunk = NamedTuple("InputChunk",
                        [("id", int),
                         ("chunk_id", int),
                         ("tokens", Tokens)])

ReferenceChunk = NamedTuple("ReferenceChunk",
                        [("id", int),
                         ("chunk_id", int),
                         ("tokens", Tokens)])
"""
A chunk of a TranslatorInput.

:param id: Sentence id.
:param chunk_id: The id of the chunk.
:param tokens: List of input tokens.
"""

TranslatorOutput = NamedTuple('TranslatorOutput', [
    ('id', int),
    ('translation', str),
    ('alignment', str),
    ('tokens', List[str]),
    ('attention_matrix', np.ndarray),
    ('score', float),
    ('coverage',np.ndarray)
])
"""
Output structure from Translator.

:param id: Id of input sentence.
:param translation: Translation string without sentence boundary tokens.
:param tokens: List of translated tokens.
:param attention_matrix: Attention matrix. Shape: (target_length, source_length).
:param score: Negative log probability of generated translation.
"""

TokenIds = List[int]
Translation = NamedTuple('Translation', [
    ('target_ids', TokenIds),
    ('attention_matrix', np.ndarray),
    ('score', float),
    ('coverage', np.ndarray),
    ('alignment', np.ndarray),
    ('source',np.ndarray)
])

TranslatedChunk = NamedTuple('TranslatedChunk', [
    ('id', int),
    ('chunk_id', int),
    ('translation', Translation),
])
"""
Translation of a chunk of a sentence.

:param id: Id of the sentence.
:param chunk_id: Id of the chunk.
:param translation: The translation of the input chunk.
"""


class ModelState:
    """
    A ModelState encapsulates information about the decoder states of an InferenceModel.
    """

    def __init__(self, states: List[mx.nd.NDArray]) -> None:
        self.states = states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray, best_hyp_pos_idx_indices: mx.nd.NDArray = None):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        #TODO (Tamer) is there a way to do mulitple indexing in mxnet without resorting to numpy?
        pos_indices =  np.array([0]) if self.states[0].shape[0] == 1 or best_hyp_pos_idx_indices is None \
                                     else   best_hyp_pos_idx_indices.asnumpy()
        for idx,state in enumerate(self.states):
            #state_np = state.asnumpy()
            #self.states[idx] = mx.nd.array(state_np[pos_indices, best_hyp_indices.asnumpy()], state.context)
            self.states[idx] = mx.nd.array(state[pos_indices, best_hyp_indices], state.context)
        #self.states = [mx.nd.take(ds, best_hyp_indices) for ds in self.states]


class LengthPenalty:
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def __call__(self, lengths: Union[mx.nd.NDArray, int, float]) -> Union[mx.nd.NDArray, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A scalar or a matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        if self.alpha == 0.0:
            if isinstance(lengths, mx.nd.NDArray):
                # no length penalty:
                return mx.nd.ones_like(lengths)
            else:
                return 1.0
        else:
            # note: we avoid unnecessary addition or pow operations
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


def _concat_translations(translations: List[Translation], start_id: int, stop_ids: Set[int],
                         length_penalty: LengthPenalty) -> Translation:
    """
    Combine translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol, attention_matrix), score and length.
    :param start_id: The EOS symbol.
    :param stop_ids: The BOS symbols.
    :return: A concatenation if the translations with a score.
    """
    # Concatenation of all target ids without BOS and EOS
    target_ids = [start_id]
    attention_matrices = []
    coverage = []
    alignment = [-1]
    source_ids = []
    offset = 0
    for idx, translation in enumerate(translations):
        assert translation.target_ids[0] == start_id
        source_ids.extend(translation.source[:])
        coverage.extend(translation.coverage[:])
        if idx == 0:
            alignment.extend([translation.alignment[1]])

        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids[1:])
            attention_matrices.append(translation.attention_matrix[1:, :])
            alignment.extend((offset + translation.alignment[2:len(translation.target_ids[:])]))
        else:
            if translation.target_ids[-1] in stop_ids:
                target_ids.extend(translation.target_ids[1:-1])
                attention_matrices.append(translation.attention_matrix[1:-1, :])
                alignment.extend((offset + translation.alignment[2:len(translation.target_ids[:])]))
            else:
                target_ids.extend(translation.target_ids[1:])
                alignment.extend(offset + translation.alignment[2:])

        offset += len(translation.source)

    # Combine attention matrices:
    attention_shapes = [attention_matrix.shape for attention_matrix in attention_matrices]
    # Adding another row for the empty BOS alignment vector
    bos_align_shape = np.asarray([1, 0])
    attention_matrix_combined = np.zeros(np.sum(np.asarray(attention_shapes), axis=0) + bos_align_shape)

    # We start at position 1 as position 0 is for the BOS, which is kept zero
    pos_t, pos_s = 1, 0
    for attention_matrix, (len_t, len_s) in zip(attention_matrices, attention_shapes):
        attention_matrix_combined[pos_t:pos_t + len_t, pos_s:pos_s + len_s] = attention_matrix
        pos_t += len_t
        pos_s += len_s

    # Unnormalize + sum and renormalize the score:
    score = sum(translation.score * length_penalty(len(translation.target_ids))
                for translation in translations)
    score = score / length_penalty(len(target_ids))
    return Translation(target_ids, attention_matrix_combined, score, coverage, alignment, source_ids)



class Translator:
    """
    Translator uses one or several models to translate input.
    It holds references to vocabularies to takes care of encoding input strings as word ids and conversion
    of target ids into a translation string.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param models: List of models.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary restriction.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 bucket_source_width: int,
                 length_penalty: LengthPenalty,
                 models: List[InferenceModel],
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int],
                 restrict_lexicon: Optional[lexicon.TopKLexicon] = None,
                 lex_weight: float = 1.,
                 align_weight: float = 0.,
                 align_skip_threshold: float = 0.,
                 align_k_best: int = 0,
                 ) -> None:
        self.context = context
        self.length_penalty = length_penalty
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.vocab_source_inv = vocab.reverse_vocab(self.vocab_source)
        self.restrict_lexicon = restrict_lexicon
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}  # type: Set[int]
        self.models = models
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.batch_size = self.models[0].batch_size
        self.lex_weight = lex_weight
        self.align_weight = align_weight
        self.dictionary = None
        self.dictionary_override_with_max_attention = False
        self.seq_idx = 0  #used with dictionaries, batching not supported
        self.align_skip_threshold = align_skip_threshold
        self.align_k_best = align_k_best
        utils.check_condition(not (self.align_skip_threshold > 0.0 and self.align_k_best > 0),
                              "Can not use threshold and histogram pruning at the same time. "
                              "--align-threshold=%s and --align-beam-size=%s" % (self.align_skip_threshold, self.align_k_best))

        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        self.max_input_length = self.models[0].max_input_length
        self.max_output_length = self.models[0].get_max_output_length(self.max_input_length)
        if bucket_source_width > 0:
            self.buckets_source = data_io.define_buckets(self.max_input_length, step=bucket_source_width)
        else:
            self.buckets_source = [self.max_input_length]
        self.pad_dist = mx.nd.full((self.batch_size * self.beam_size, len(self.vocab_target)), val=np.inf,
                                   ctx=self.context)

        logger.info("Translator (%d model(s) beam_size=%d ensemble_mode=%s batch_size=%d "
                    "buckets_source=%s)",
                    len(self.models),
                    self.beam_size,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.batch_size,
                    self.buckets_source)

    @staticmethod
    def _get_interpolation_func(ensemble_mode):
        if ensemble_mode == 'linear':
            return Translator._linear_interpolation
        elif ensemble_mode == 'log_linear':
            return Translator._log_linear_interpolation
        else:
            raise ValueError("unknown interpolation type")

    @staticmethod
    def _linear_interpolation(predictions):
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([mx.nd.log(p) for p in predictions])
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(mx.nd.softmax(log_probs))

    @staticmethod
    def make_input(sentence_id: int, sentence: str, reference: str) -> TranslatorInput:
        """
        Returns TranslatorInput from input_string

        :param sentence_id: Input sentence id.
        :param sentence: Input sentence.
        :param sentence: Reference (target) sentence.
        :return: Input for translate method.
        """
        tokens = list(data_io.get_tokens(sentence))
        reference_tokens = list(data_io.get_tokens(reference)) if reference else []
        return TranslatorInput(id=sentence_id, sentence=sentence.rstrip(), tokens=tokens,
                               reference_tokens=reference_tokens)

    def translate(self, trans_inputs: List[TranslatorInput]) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Splits oversized sentences to sentence chunks of size less than max_input_length.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :return: List of translation results.
        """
        translated_chunks = []

        # split into chunks
        input_chunks = [] # type: List[InputChunk]
        reference_chunks = []  # type: List[ReferenceChunk]
        for input_idx, trans_input in enumerate(trans_inputs):

            if len(trans_input.tokens) > 0 and trans_input.reference_tokens:
                if len(trans_input.reference_tokens) - len(trans_input.tokens) >= C.MAX_JUMP:
                    logger.warning(
                        "Reference %d has length (%d) that cant be aligned with input length (%d). "
                        "Removing sentence from corpus",
                        trans_input.id, len(trans_input.reference_tokens), len(trans_input.tokens))
                    empty_translation = Translation(target_ids=[],
                                                    attention_matrix=np.asarray([[0]]),
                                                    score=-np.inf,
                                                    coverage=np.asarray([]),
                                                    source=[],
                                                    alignment=np.asarray([-1]))
                    translated_chunks.append(TranslatedChunk(id=input_idx,
                                                             chunk_id=0,
                                                             translation=empty_translation))
                    continue

            if len(trans_input.tokens) == 0:
                empty_translation = Translation(target_ids=[],
                                                attention_matrix=np.asarray([[0]]),
                                                score=-np.inf,
                                                coverage=np.asarray([]),
                                                source=[],
                                                alignment=np.asarray([-1]))
                translated_chunks.append(TranslatedChunk(id=input_idx,
                                                         chunk_id=0,
                                                         translation=empty_translation))
            elif len(trans_input.tokens) >= self.max_input_length:
                logger.debug(
                    "Input %d has length (%d) that exceeds max input length (%d). Splitting into chunks of size %d.",
                    trans_input.id, len(trans_input.tokens), self.buckets_source[-1], self.max_input_length)
                #
                # split into chunks of size max_input_length - 1 to account for additional EOS token
                # TODO (gabriel) verify
                #
                token_chunks = utils.chunks(trans_input.tokens, self.max_input_length - 1)
                input_chunks.extend(InputChunk(input_idx, chunk_id, chunk)
                                    for chunk_id, chunk in enumerate(token_chunks))
                if trans_input.reference_tokens:
                    # TODO (gabriel) split reference token to the same length as input string
                    # this splitting is kind of arbitrary as it assumes a one-to-one alignment
                    ref_token_chunks = utils.chunks(trans_input.reference_tokens, self.max_input_length - 1)
                    reference_chunks.extend(ReferenceChunk(input_idx, chunk_id, chunk)
                                    for chunk_id, chunk in enumerate(ref_token_chunks))

            else:
                input_chunks.append(InputChunk(input_idx, 0, trans_input.tokens))
                if trans_input.reference_tokens:
                    reference_chunks.append(ReferenceChunk(input_idx, 0, trans_input.reference_tokens))

        if len(input_chunks) == 0:
            logger.warning("no sentences left for translation")
            return []

        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks,reference_chunks  = (list(t) for t in zip(*sorted(itertools.zip_longest(input_chunks,reference_chunks),
                                                key=lambda args : len(args[0].tokens),
                                                reverse=True)))
        # translate in batch-sized blocks over input chunks

        for batch_id, (chunks, ref_chunks) in enumerate(itertools.zip_longest(utils.grouper(input_chunks, self.batch_size),
                                                                utils.grouper(reference_chunks, self.batch_size)
                                                                    if reference_chunks is not None else [None])):
            batch = [chunk.tokens for chunk in chunks]
            reference_batch = [chunk.tokens for chunk in ref_chunks] if len(ref_chunks)>0 \
                                                                        and ref_chunks[0] is not None else []
            logger.debug("Translating batch %d", batch_id)
            # underfilled batch will be filled to a full batch size with copies of the 1st input
            rest = self.batch_size - len(batch)
            if rest > 0:
                logger.debug("Extending the last batch to the full batch size (%d)", self.batch_size)
                batch = batch + [batch[0]] * rest
                if len(ref_chunks) > 0 and ref_chunks[0] is not None:
                    reference_batch = reference_batch + [reference_batch[0]] * rest
            batch_translations = self.translate_nd(*self._get_inference_input(batch,reference_batch),batch)

            self.seq_idx += 1
            # truncate to remove filler translations
            if rest > 0:
                batch_translations = batch_translations[:-rest]

            for chunk, translation in zip(chunks, batch_translations):
                translated_chunks.append(TranslatedChunk(chunk.id, chunk.chunk_id, translation))
        # Sort by input idx and then chunk id
        translated_chunks = sorted(translated_chunks)

        # Concatenate results
        results = []
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.id)
        for trans_input, (input_idx, chunks) in zip(trans_inputs, chunks_by_input_idx):
            chunks = list(chunks)
            if len(chunks) == 1:
                translation = chunks[0].translation
            else:
                translations_to_concat = [translated_chunk.translation for translated_chunk in chunks]
                translation = self._concat_translations(translations_to_concat)

            results.append(self._make_result(trans_input, translation))

        return results

    def _get_inference_input(self, sequences: List[List[str]], reference_sequences: List[List[str]] ) -> Tuple[mx.nd.NDArray, int]:
        """
        Returns NDArray of source ids (shape=(batch_size, bucket_key)) and corresponding bucket_key.

        :param sequences: List of lists of input tokens.
        :param sequences: List of lists of reference tokens.
        :return NDArray of source ids, bucket key, source lengths, and reference ids
        """
        #+1 for EOS
        bucket_key = data_io.get_bucket(max(len(tokens)+1 for tokens in sequences), self.buckets_source)

        utils.check_condition(C.PAD_ID == 0, "pad id should be 0")
        source = mx.nd.zeros((len(sequences), bucket_key))
        # max reference length is longest reference in batch + 1 for EOS
        max_reference_length = max([len(x) for x in reference_sequences]) + 1 if reference_sequences else 0
        reference = mx.nd.zeros((len(reference_sequences), max_reference_length)) if reference_sequences else []
        source_length = []
        for j, (tokens, reference_tokens) in enumerate(itertools.zip_longest(sequences,reference_sequences)):
            ids = data_io.tokens2ids(tokens, self.vocab_source, has_categ_content=True)
            reference_ids = data_io.tokens2ids(reference_tokens,
                                     self.vocab_target, has_categ_content=True) if reference_tokens else []
            for i, wid in enumerate(ids):
                source[j, i] = wid
            source[j,len(tokens)] = self.vocab_source[C.EOS_SYMBOL]
            #source_length needed for alignment-based models, where batch_size=len(sequences)=1
            source_length.append(len(tokens) + 1) # +1 for EOS

            for i, wid in enumerate(reference_ids):
                reference[j, i] = wid
            if reference_sequences:
                reference[j,len(reference_tokens)] = self.vocab_target[C.EOS_SYMBOL]

        return source, bucket_key, source_length, reference

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        # remove special sentence start symbol (<s>) from the output:
        target_ids = translation.target_ids[1:]
        attention_matrix = translation.attention_matrix[1:, :]

        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]

        if np.max(translation.alignment) > -1:
            target_tokens = [token + '_' + translation.source[translation.alignment[i+1]]
                             if (translation.alignment[i+1]<len(translation.source) and
                                 (token == C.UNK_SYMBOL or token == C.NUM_SYMBOL
                                  or token == C.NUM_SYMBOL_2)) else token
                             for i, token in enumerate(target_tokens[:-1])] + [target_tokens[-1]]
        else:
            target_tokens = [token + '_' + translation.source[np.argmax(translation.attention_matrix[i+1])]
                                if (token == C.UNK_SYMBOL or token == C.NUM_SYMBOL
                                    or token == C.NUM_SYMBOL_2) and \
                                    np.argmax(translation.attention_matrix[i + 1]) < len(translation.source) else token
                             for i,token in enumerate(target_tokens) ]
        target_string = C.TOKEN_SEPARATOR.join(
            target_token for target_id, target_token in zip(target_ids, target_tokens) if
            target_id not in self.stop_ids)
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]
        coverage = translation.coverage[:]

        return TranslatorOutput(id=trans_input.id,
                                translation=target_string,
                                alignment=translation.alignment,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=translation.score,
                                coverage=coverage)

    def _concat_translations(self, translations: List[Translation]) -> Translation:
        """
        Combine translations through concatenation.

        :param translations: A list of translations (sequence, attention_matrix), score and length.
        :return: A concatenation if the translations with a score.
        """
        return _concat_translations(translations, self.start_id, self.stop_ids, self.length_penalty)

    def translate_nd(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     actual_source_length: List[int],
                     reference: mx.nd.NDArray,
                     original_source: List[List[str]]) -> List[Translation]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (batch_size, bucket_key).
        :param source_length: Bucket key.
        :param actual_source_length: source_length_without padding
                :param reference: Reference ids. Shape: (batch_size, max_output_length).

        :param original_source: original source words before vocabulary mappying (batch_size,)

        :return: Sequence of translations.
        """
        return self._get_best_from_beam(*self._beam_search(source, source_length, actual_source_length,reference),original_source)

    def _encode(self, sources: mx.nd.NDArray, source_length: int) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param sources: Source ids. Shape: (batch_size, bucket_key).
        :param source_length: Bucket key.
        :return: List of ModelStates.
        """
        #add bos and remove last element to keep the length
        #bos_sources = mx.ndarray.concat(mx.ndarray.array([[self.vocab_source[C.BOS_SYMBOL]]] *self.batch_size,
        #                                                 ctx=sources.context)
        #                                , sources[:,:sources.shape[1]-1], dim=1)
        bos_sources = mx.ndarray.concat(mx.ndarray.array([[self.vocab_source[C.BOS_SYMBOL]]] * self.batch_size,
                                                         ctx=sources.context)
                                        , sources[:, :sources.shape[1]], dim=1)
        return [model.run_encoder( bos_sources
                                           if model.alignment_model
                                           else sources,
                                   source_length + (1 if model.alignment_model else 0)) for model in self.models]

    @staticmethod
    def _relative_jump_to_abs_alignment(last_alignment, alignments):
        return (alignments - (C.NUM_ALIGNMENT_JUMPS -1)/2 + last_alignment).astype('int32')

    def _decode_step(self,
                     sequences: mx.nd.NDArray,
                     step: int,
                     source_length: int,
                     actual_soruce_length: int,
                     max_output_length: int,
                     states: List[ModelState],
                     models_output_layer_w: List[mx.nd.NDArray],
                     models_output_layer_b: List[mx.nd.NDArray],
                     prev_alignment: mx.nd.NDArray,
                     coverage_vector: mx.nd.NDArray,
                     last_alignment: mx.nd.NDArray,
                     previous_jump: mx.nd.NDArray) \
            -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState] ]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param sequences: Sequences of current hypotheses. Shape: (batch_size * beam_size, max_output_length).
        :param step: Beam search iteration.
        :param source_length: Length of the input sequence.
        :param actual_soruce_length: actual source length without padding
        :param max_output_length: Maximum output length.
        :param states: List of model states.
        :param models_output_layer_w: Custom model weights for logit computation (empty for none).
        :param models_output_layer_b: Custom model biases for logit computation (empty for none).
        :param prev_alignment: last aligned source positions
        :param coverage_vector (batch* beam_size, source_length) coverage vector for each beam entry
        :param last_alignment: (batch* beam_size,1) last aligned source positions
        :param previous_jump: (batch* beam_size,1) source jump
        :return: (probs, attention scores, list of model states)
        """
        bucket_key = (source_length, step)
        prev_word = sequences[:, step - 1]


        model_probs, model_attention_probs, model_states = [], [], [None]*len(self.models)

        # alignment models
        has_align_model = False
        align_model_probs, align_model_states = [], []
        num_align_models = 0
        for model_idx, (model, state) in enumerate(itertools.zip_longest(self.models, states)):
            # alignment models evaluated elsewhere
            if not model.alignment_model:
                continue
            num_align_models += 1
            has_align_model = True
            probs, _, state, _ = model.run_decoder(prev_word=prev_word,
                                                   bucket_key=bucket_key,
                                                   model_state=state,
                                                   prev_alignment=prev_alignment,
                                                   step=step,
                                                   last_alignment=last_alignment,
                                                   previous_jump=previous_jump,
                                                   actual_source_length=actual_soruce_length,
                                                   use_unaligned=self.use_unaligned)
            align_model_probs.append(probs)
            model_states[model_idx] = state

        skip_alignments = self.calculate_skip_alignment_list(actual_soruce_length, align_model_probs, bucket_key,
                                                             last_alignment, num_align_models, step)
        # We use zip_longest here since we'll have empty lists when not using restrict_lexicon
        #lexical models
        for model_idx, (model, out_w, out_b, state) in enumerate(itertools.zip_longest(
                self.models, models_output_layer_w, models_output_layer_b, states)):
            #alignment models evaluated later
            if model.alignment_model:
                continue
            decoder_outputs, attention_probs, state, new_alignment = model.run_decoder(prev_word, bucket_key,
                                                                                       state,
                                                                                       step=step,
                                                                                       actual_source_length=actual_soruce_length,
                                                                                       use_unaligned=self.use_unaligned,
                                                                                       skip_alignments=skip_alignments)
            # Compute logits and softmax with restricted vocabulary
            if self.restrict_lexicon:
                logits = model.output_layer(decoder_outputs, out_w, out_b)
                probs = mx.nd.softmax(logits)
            else:
                # Otherwise decoder outputs are already target vocab probs
                probs = decoder_outputs
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states[model_idx] = state

        neg_logprobs , attention_probs = self._combine_predictions_per_position(model_probs,model_attention_probs)

        if has_align_model:
            align_neg_logprobs, _ = self._combine_predictions_per_position(align_model_probs) # Alignment mo
            neg_logprobs = self._combine_lex_align_scores(align_neg_logprobs, neg_logprobs,
                                                          prev_alignment, new_alignment, coverage_vector,
                                                          actual_soruce_length, last_alignment)
        return neg_logprobs, attention_probs, model_states

    def calculate_skip_alignment_list(self, actual_source_length, align_model_probs, bucket_key, last_alignment,
                                      num_align_models, step):
        """
        Calculate list of alignment points to prune
        :param actual_source_length:
        :param align_model_probs:
        :param bucket_key:
        :param last_alignment:
        :param num_align_models:
        :param step:
        :return:
        """
        skip_alignments = []
        start = mx.nd.clip((C.NUM_ALIGNMENT_JUMPS - 1) / 2 - last_alignment, 0, C.NUM_ALIGNMENT_JUMPS).reshape(
            (-1,))
        end = mx.nd.clip(start + bucket_key[0], 0, C.NUM_ALIGNMENT_JUMPS).reshape((-1,))

        if self.align_skip_threshold > 0:
            skip_alignments = self._alignment_threshold_pruning(self.align_skip_threshold, align_model_probs, bucket_key,
                                                                num_align_models, start, end)
        elif self.align_k_best > 0:
            skip_alignments = self._alignment_histogram_pruning(self.align_k_best, align_model_probs, bucket_key,
                                                                num_align_models, start, end)

        if len(skip_alignments) > 0:
            alignment_end_idx = max(0, min(C.MAX_JUMP, max(actual_source_length) - step) + min(C.MAX_JUMP, step - 1)) + 1
            if np.all(skip_alignments[align_idx_offset(step):alignment_end_idx + align_idx_offset(step)]):
                num_skipped_alignments = 0
                skipped_alignments_string = "all"
            else:
                skipped_alignments_string = ", ".join([str(i) for i, x in enumerate(skip_alignments) if x])
                num_skipped_alignments = np.sum(skip_alignments)
        else:
            num_skipped_alignments = 0
            skipped_alignments_string = ""

        logger.info("num skipped alignments %d [%s]" % (num_skipped_alignments, skipped_alignments_string))

        return skip_alignments

    def _alignment_histogram_pruning(self, align_beam_size, align_model_probs, bucket_key,
                                     num_align_models, start, end):
        """
        calculate list of alignment points to prune by keeping the top-k alingment points per hypothesis
        :param align_model_probs:
        :param bucket_key:
        :param num_align_models:
        :return:
        """
        utils.check_condition(num_align_models == 1, "Skip alignments only implemented for one alignment model")
        skip_alignments = np.array([True] * bucket_key[0])
        np_align_model_probs = align_model_probs[0].asnumpy()
        for sent in range(self.batch_size * self.beam_size):
            source_sel = slice(start[sent].asscalar(), end[sent].asscalar())
            rows = slice(sent, (sent + 1))
            sliced_scores = np_align_model_probs[:, rows, source_sel]  # .reshape(shape=(1, -1))
            if (source_sel.stop - source_sel.start) > align_beam_size:
                # returns: best_hyp_indices_, best_hyp_pos_indices , best_word_indices
                (_, _, best_word_indices), _ = utils.smallest_k( -1 * sliced_scores, align_beam_size,  False)
                for k in best_word_indices:
                    if 0 <= k < bucket_key[0]:
                        skip_alignments[k] = False
            else:
                for k in range(source_sel.stop - source_sel.start):
                    skip_alignments[k] = False

        return skip_alignments

    def _alignment_threshold_pruning(self, align_skip_threshold, align_model_probs, bucket_key,
                                     num_align_models, start, end):
        """
        calculating list of alignment points to prune which do not reach a threshold
        :param align_model_probs:
        :param bucket_key:
        :param last_alignment:
        :param num_align_models:
        :param step:
        :return:
        """

        utils.check_condition(num_align_models == 1, "Skip alignments only implemented for one alignment model")
        skip_jumps = mx.nd.zeros((self.batch_size * self.beam_size, bucket_key[0]))

        for idx in range(self.batch_size * self.beam_size):
            source_sel = slice(start[idx].asscalar(), end[idx].asscalar())
            target_sel = slice(0, source_sel.stop - source_sel.start)
            skip_jumps[idx, target_sel] = align_model_probs[0][0, idx, source_sel]

        skip_alignments = np.all((skip_jumps < align_skip_threshold).asnumpy(), axis=0)

        return skip_alignments

    def _combine_lex_align_scores(self,
                                  align_neg_logprobs: mx.nd.NDArray,
                                  lex_neg_logprobs: mx.nd.NDArray,
                                  prev_alignment: mx.nd.NDArray,
                                  new_alignment: mx.nd.NDArray,
                                  coverage_vector: mx.ndarray.NDArray,
                                  actual_soruce_length: int,
                                  last_alignment: mx.ndarray.NDArray) ->  mx.nd.NDArray:


        """
        Returns combined lexical and alignment negative log scores

        :param align_neg_logprobs:  Shape(source_length, beam_size, C.NUM_ALIGNMENT_JUMPS).
        :param lex_neg_logprobs: List of Shape(1, beam_size, target_vocab_size).
        :param prev_alignment: Shape(beam_size,1)
        :param new_alignment: Shape(source_length, beam_size, source_length)
        :param coverage_vector (batch* beam_size, source_length) coverage vector for each beam entry
        :param actual_soruce_length: actual source length without padding
        :param last_alignment: (batch* beam_size,1) last aligned source positions
        :return: Combined weighted negative log probabilities
        """
        combined_result = lex_neg_logprobs
        alignment_jump_idx = None
        for j in range(new_alignment.shape[0]):
            disallowed_alignments = self._invalid_alignments(prev_alignment,
                                                            new_alignment[j][0],
                                                            coverage_vector)
            alignment_jump_idx =  C.UNALIGNED_JUMP_LABEL * \
                                        mx.ndarray.ones_like(data=last_alignment, ctx=self.context, dtype='int32') \
                                        if new_alignment[j][0] == C.UNALIGNED_SOURCE_INDEX and self.use_unaligned \
                                                else   (C.NUM_ALIGNMENT_JUMPS-1)/2 + new_alignment[j][0] - last_alignment
            #jump_scores = mx.ndarray.batch_take(align_neg_logprobs[0],alignment_jump_idx)
            jump_scores = mx.ndarray.pick(align_neg_logprobs[0], alignment_jump_idx, keepdims=True)

            #combined_result[j] = self.lex_weight * lex_neg_logprobs[j] + self.align_weight * jump_scores
            combined_result[j] = mx.nd.where(mx.nd.split(disallowed_alignments,num_outputs=1,squeeze_axis=1),
                                             mx.nd.ones(shape=lex_neg_logprobs[j].shape,ctx=self.context)*np.inf,
                                             self.lex_weight * lex_neg_logprobs[j] + self.align_weight * jump_scores)
            #enforce sentence-end to sentence-end alignment
            for sent in range(len(actual_soruce_length)):
                if new_alignment[j][0] != actual_soruce_length[sent] -1 and \
                        new_alignment[j][0] != C.UNALIGNED_SOURCE_INDEX:
                    combined_result[j,sent*self.beam_size:(sent+1)*self.beam_size,self.vocab_target[C.EOS_SYMBOL]] = np.inf

        return combined_result

    def _invalid_alignments(self,
                            prev_alignment: mx.nd.NDArray,
                            new_alignment: mx.nd.NDArray,
                            coverage_vector: mx.nd.NDArray):
        """
        Returns List of invalid alignments that violate max jump and coverage constraints

        :param prev_alignment: Shape(beam_size,1)
        :param new_alignment: Shape(1,)
        :param coverage_vector (batch* beam_size, source_length) coverage vector for each beam entry
        :return: List of invalid alignment indices within the beam
        """

        #unaligned positions produce no violations
        if new_alignment == C.UNALIGNED_SOURCE_INDEX and self.use_unaligned:
            return mx.nd.zeros(ctx=self.context,shape=(self.beam_size*self.batch_size,1))

        max_jump = C.MAX_JUMP
        jump = new_alignment - prev_alignment
        #coverage_violation = coverage_vector + mx.ndarray.one_hot(new_alignment*mx.ndarray.ones(ctx=self.context,
        #                                                                                        shape=(coverage_vector.shape[0])
        #                                                                                        ,dtype='int32'),dtype='int32',
        #                                                          depth=10)
        coverage_slice = mx.ndarray.pick(coverage_vector, mx.ndarray.ones(shape=(self.beam_size*self.batch_size,1), ctx=self.context,
                                                         dtype='int32') * new_alignment,
                                         keepdims=True)
        coverage_violation = coverage_slice + 1> C.MAX_COVERAGE
        ret = mx.nd.where(coverage_violation > 0,
                          mx.nd.ones(ctx=self.context,shape=(self.beam_size*self.batch_size,1)),
                          mx.nd.zeros(ctx=self.context,shape=(self.beam_size*self.batch_size,1)))
        return ret

    def _combine_predictions_per_position(self,
                             probs: List[List[mx.nd.NDArray]],
                             attention_probs: List[List[mx.nd.NDArray]] = None) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined position-dependent predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: Model List of Source Position List of Shape(beam_size, target_vocab_size).
        :param attention_probs: Model List of Position List of Shape(beam_size, bucket_key).
        :return: Model-combined per-position negative log probabilities, averaged attention scores.
        """
        combined_neg_logprobs, combined_attention_probs = None, None
        for j in range(probs[0].shape[0]):
            model_probs_j = [probs[m][j] for m in range(len(probs))]
            model_attention_probs_j = None
            if attention_probs is not None:
                model_attention_probs_j = [attention_probs[m][j] for m in range(len(attention_probs))]
            combined_neg_logprobs_j, combined_attention_probs_j = self._combine_predictions(model_probs_j,
                                                                                                    model_attention_probs_j)
            if combined_neg_logprobs is None:
                combined_neg_logprobs = mx.ndarray.zeros(ctx=self.context,shape=(probs[0].shape[0],combined_neg_logprobs_j.shape[0],
                                                                               combined_neg_logprobs_j.shape[1]),dtype='float32')
            combined_neg_logprobs[j, :, :] = combined_neg_logprobs_j
            if combined_attention_probs_j is not None:
                if combined_attention_probs is None:
                    combined_attention_probs = mx.ndarray.zeros(ctx=self.context,shape=(probs[0].shape[0],combined_attention_probs_j.shape[0],
                                                                                        combined_attention_probs_j.shape[1]),dtype='float32')
                combined_attention_probs[j,:,:] = combined_attention_probs_j

        return combined_neg_logprobs, combined_attention_probs

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray] = None) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined negative log probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = None
        if attention_probs is not None:
            attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(probs[0])  # pylint: disable=invalid-unary-operand-type
        else:
            neg_logprobs = self.interpolation_func(probs)
        return neg_logprobs, attention_prob_score

    def _override_scores_with_dictionary(self,scores,attention_scores,source,t,alignment_based):
        if alignment_based:
            if self.dictionary_override_with_max_attention:
                max_attention = mx.ndarray.argmax(attention_scores[:scores.shape[0]], axis=2)
                max_attention_cpu = max_attention.copyto(source.context)
                #new_source = mx.nd.expand_dims(source, axis=1).broadcast_to(
                #    (max_attention_cpu.shape[0], scores.shape[1], source.shape[1]))
                #(batch_size * num_src_hyp_pos, source_len)
                #new_source = mx.nd.reshape(new_source,shape=(-3,0))
                #max_attention_source_words = mx.nd.batch_take(
                #    new_source,
                #    max_attention_cpu.astype(dtype="int32")).asnumpy()
                for beam_idx in range (0,max_attention_cpu.shape[0]):
                    for idx,src_pos in enumerate(max_attention_cpu[beam_idx]):
                        source_word = mx.nd.take(source[0],src_pos.astype(dtype="int32"))
                        #src_pos = src_pos.astype(dtype="int32").asscalar() -align_idx_offset(t) +\
                        #          (1 if self.use_unaligned else 0)
                        source_word_str = self.vocab_source_inv[int(source_word.asscalar())]
                        if source_word_str in self.dictionary[self.seq_idx]:
                            target_word_str = self.dictionary[self.seq_idx][source_word_str]
                            if target_word_str in self.vocab_target:
                                target_word = self.vocab_target[target_word_str]
                                target_word_score = scores[beam_idx, idx, target_word]
                                scores[beam_idx, idx, :] = np.inf
                                scores[beam_idx, idx, target_word] = target_word_score
            else:
                for j in range(0, scores.shape[1]):
                    src_pos = mx.nd.array([j + align_idx_offset(t) - (1 if self.use_unaligned else 0)],dtype="int32")
                    source_word = mx.nd.pick(source, src_pos).asnumpy()
                    source_word_str = self.vocab_source_inv[int(source_word)]
                    if self.seq_idx in self.dictionary and source_word_str in self.dictionary[self.seq_idx]:
                        target_word_str = self.dictionary[self.seq_idx][source_word_str]
                        if target_word_str in self.vocab_target:
                            target_word = self.vocab_target[target_word_str]
                            target_word_scores = scores[:, j, target_word]
                            scores[:, j, :] = np.inf
                            scores[:, j, target_word] = target_word_scores

        else:
            max_attention = mx.ndarray.argmax(attention_scores[:scores.shape[0]], axis=2)
            max_attention_cpu = max_attention.copyto(source.context)
            max_attention_source_words = mx.nd.pick(
                source.broadcast_to((max_attention_cpu.shape[0], source.shape[1])),
                max_attention_cpu.astype(dtype="int32")).asnumpy()
            for beam_idx, word in enumerate(max_attention_source_words):
                source_word_str = self.vocab_source_inv[int(word)]
                if self.seq_idx in self.dictionary and source_word_str in self.dictionary[self.seq_idx]:
                    target_word_str = self.dictionary[self.seq_idx][source_word_str]
                    if target_word_str in self.vocab_target:
                        target_word = self.vocab_target[target_word_str]
                        target_word_score = scores[beam_idx, 0, target_word]
                        scores[beam_idx, 0, :] = np.inf
                        scores[beam_idx, 0, target_word] = target_word_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     actual_source_length: List[int],
                     reference: mx.nd.NDArray) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, \
                                                         mx.nd.NDArray, mx.nd.NDArray]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key).
        :param source_length: Max source length.
        :param actual_source_length: Source length without padding
        :param reference: Reference ids. Shape: (batch_size, max_output_length).
        :return List of lists of word ids, list of attentions, array of accumulated length-normalized
                negative log-probs, lengths, coverage vector, alignments
        """
        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length) \
            if reference is None or len(reference) == 0 else reference.shape[1]+1

        # General data structure: each row has batch_size * beam blocks for the 1st sentence, with a full beam,
        # then the next block for the 2nd sentence and so on

        # sequences: (batch_size * beam_size, output_length), pre-filled with <s> symbols on index 0
        sequences = mx.nd.full((self.batch_size * self.beam_size, max_output_length), val=C.PAD_ID, ctx=self.context,
                               dtype='int32')
        sequences[:, 0] = self.start_id

        lengths = mx.nd.ones((self.batch_size * self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # attentions: (batch_size * beam_size, output_length, encoded_source_length)
        attentions = mx.nd.zeros((self.batch_size * self.beam_size, max_output_length, encoded_source_length),
                                 ctx=self.context)

        # best_hyp_indices: row indices of smallest scores (ascending).
        best_hyp_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # best_hyp_pos_idx_indices: related to alignment-based NMT: source position indices of smallest scores (ascending).
        best_hyp_pos_idx_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # best_hyp_pos_indices: related to alignment-based NMT: source
        # alignment possible including unaligned position indices as defined in constants.py
        best_hyp_pos_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # best_word_indices: column indices of smallest scores (ascending).
        best_word_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.batch_size * self.beam_size, 1, 1), ctx=self.context)
        # coverage vectors
        coverage_vector = mx.nd.zeros((self.batch_size * self.beam_size,source_length), ctx=self.context, dtype='int32')

        # best_hyp_indices_np = np.empty((self.batch_size * self.beam_size,), dtype='int32')
        best_hyp_indices_mx = mx.nd.empty((self.batch_size * self.beam_size,), dtype='int32')
        # best_hyp_pos_indices_np = np.empty((self.batch_size * self.beam_size,), dtype='int32')
        best_hyp_pos_indices_mx = mx.nd.empty((self.batch_size * self.beam_size,), dtype='int32')
        # best_word_indices_np = np.empty((self.batch_size * self.beam_size,), dtype='int32')
        best_word_indices_mx = mx.nd.empty((self.batch_size * self.beam_size,), dtype='int32')
        # scores_accumulated_np = np.empty((self.batch_size * self.beam_size,))
        scores_accumulated_mx = mx.nd.empty((self.batch_size * self.beam_size,))
        # attention_scores_np = np.empty((self.batch_size * self.beam_size,encoded_source_length,encoded_source_length))

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        models_output_layer_w = list()
        models_output_layer_b = list()
        pad_dist = self.pad_dist
        #TODO (Tamer) remove instate from init?
        pad_dist = mx.nd.full((self.batch_size * self.beam_size,
                                          2*C.MAX_JUMP + 2 if self.models[0].alignment_based else 1,
                                          len(self.vocab_target)),
                                  val=np.inf, ctx=self.context)
        vocab_slice_ids = None  # type: mx.nd.NDArray
        if self.restrict_lexicon:
            # TODO: See note in method about migrating to pure MXNet when set operations are supported.
            #       We currently convert source to NumPy and target ids back to NDArray.
            vocab_slice_ids = mx.nd.array(self.restrict_lexicon.get_trg_ids(source.astype("int32").asnumpy()),
                                          ctx=self.context)

            if vocab_slice_ids.shape[0] < self.beam_size + 1:
                # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
                # smaller than the beam size.
                logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                               vocab_slice_ids.shape[0], self.beam_size)
                n = self.beam_size - vocab_slice_ids.shape[0] + 1
                vocab_slice_ids = mx.nd.concat(vocab_slice_ids,
                                               mx.nd.full((n,), val=self.vocab_target[C.EOS_SYMBOL], ctx=self.context),
                                               dim=0)

            pad_dist = mx.nd.full((self.batch_size * self.beam_size, 1, vocab_slice_ids.shape[0]),
                                  val=np.inf, ctx=self.context)
            for m in self.models:
                models_output_layer_w.append(m.output_layer_w.take(vocab_slice_ids))
                models_output_layer_b.append(m.output_layer_b.take(vocab_slice_ids))

        # (0) encode source sentence, returns a list
        model_states = self._encode(source, source_length)

        #initial alignments set to -1
        alignment = mx.nd.zeros(ctx=self.context,shape=(self.batch_size * self.beam_size,max_output_length),dtype='int32') -1
        alignment_jump = mx.nd.zeros(ctx=self.context,shape=(self.batch_size * self.beam_size,1), dtype='int32')
        #keep track of last aligned position needed for handling jumps after unaligned target words
        last_aligned = mx.nd.zeros(ctx=self.context,shape=(self.batch_size * self.beam_size,),dtype='int32') -1
        alignment_based = any([model.alignment_based for model in self.models])
        self.use_unaligned = any([model.use_unaligned for model in self.models])
        logger.info("source length: %s",' '.join(str(i) for i in actual_source_length))
        alignment_jump_offset = (C.NUM_ALIGNMENT_JUMPS - 1) / 2
        for t in range(1, max_output_length):

            # (1) obtain next predictions and advance models' state
            # scores: (batch_size * beam_size, target_vocab_size)
            # attention_scores: (batch_size * beam_size, bucket_key)
            scores, attention_scores, model_states = self._decode_step(sequences,
                                                                       t,
                                                                       source_length,
                                                                       actual_source_length,
                                                                       max_output_length,
                                                                       model_states,
                                                                       models_output_layer_w,
                                                                       models_output_layer_b,
                                                                       prev_alignment=mx.nd.slice_axis(alignment,axis=1,begin=t-1,end=t),
                                                                       coverage_vector=coverage_vector,
                                                                       last_alignment=mx.nd.expand_dims(data=last_aligned, axis=1),
                                                                       previous_jump=alignment_jump)
            # TODO remove for performance
            scores = mx.ndarray.swapaxes(scores,0,1)
            attention_scores = mx.ndarray.swapaxes(attention_scores,0,1)

            if alignment_based:
                active_positions = slice(0, min(
                    min(C.MAX_JUMP, max(actual_source_length) - t) + min(C.MAX_JUMP, t - 1) + 1,
                    max(actual_source_length)
                ))
                logger.info("active posititions %s, actual source lengths %s, target position %d" % (
                active_positions, actual_source_length, t))
            else:
                active_positions = slice(0, 1)

            # (2) compute length-normalized accumulated scores in place
            if t == 1 and self.batch_size == 1:  # only one hypothesis at t==1
                scores = scores[:1, active_positions] / self.length_penalty(lengths[:1])
            else:
                # renormalize scores by length ...
                scores = (scores + scores_accumulated * mx.nd.expand_dims(self.length_penalty(lengths - 1),axis=1)) / mx.nd.expand_dims(self.length_penalty(lengths),axis=1)
                # ... but not for finished hyps.
                # their predicted distribution is set to their accumulated scores at C.PAD_ID.
                #TODO (Tamer) Verify!
                pad_dist[:, :,C.PAD_ID] = scores_accumulated[:,:,0].broadcast_to((pad_dist.shape[0],pad_dist.shape[1]))
                # this is equivalent to doing this in numpy:
                #   pad_dist[finished, :] = np.inf
                #   pad_dist[finished, C.PAD_ID] = scores_accumulated[finished]

                if active_positions == slice(0,0):
                    break
                scores = mx.nd.where(finished, pad_dist[:, active_positions, :], scores)

            # (3) get beam_size winning hypotheses for each sentence block separately
            # TODO(fhieber): once mx.nd.topk is sped-up no numpy conversion necessary anymore.
            # scores = scores.asnumpy()  # convert to numpy once to minimize cross-device copying
            #####
            if self.dictionary:
                self._override_scores_with_dictionary(scores,attention_scores,source,t,alignment_based)
            #DEBUGGING
            #score_wish = []
            #score_wish[:] = scores[:,9,306]
            #scores[:,9,:] = np.inf
            #scores[:, 9, 306] = score_wish


            best_hyp_indices, best_hyp_pos_idx_indices, best_word_indices, scores_accumulated = self.topk(
                actual_source_length,
                alignment_based,
                reference,
                scores,
                t)


            #best_hyp_indices[:] = best_hyp_indices_mx
            offset = align_idx_offset(t)
            best_hyp_pos_indices[:] = best_hyp_pos_idx_indices + offset - (1 if self.use_unaligned else 0)
            #best_hyp_pos_indices[:] = best_hyp_pos_indices_mx + offset - (1 if self.use_unaligned else 0)
            #best_hyp_pos_idx_indices[:] = best_hyp_pos_indices_mx

            #best_word_indices[:] = best_word_indices_mx
            scores_accumulated = mx.nd.expand_dims(mx.nd.expand_dims(scores_accumulated, axis=1),axis=1)
            # Map from restricted to full vocab ids if needed
            if self.restrict_lexicon:
                best_word_indices[:] = vocab_slice_ids.take(best_word_indices)

            # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending)
            sequences = mx.nd.take(sequences, best_hyp_indices)
            lengths = mx.nd.take(lengths, best_hyp_indices)
            finished = mx.nd.take(finished, best_hyp_indices)
            #attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
            # attention_scores_np = attention_scores.asnumpy()
            attentions = mx.nd.take(attentions, best_hyp_indices)

            # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
            # pylint: disable=unsupported-assignment-operation
            sequences[:, t] = best_word_indices
            attentions[:, t, :] = attention_scores[best_hyp_indices,best_hyp_pos_idx_indices,:]
            #attentions[:, t, :] = attention_scores
            lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')

            #(6) update coverage vector of active hypotheses
            coverage_vector = mx.nd.take(coverage_vector,best_hyp_indices) +\
                              mx.nd.where(finished,
                                          mx.nd.zeros(ctx=self.context,dtype='int32',shape=(self.beam_size* self.batch_size,source_length)),
                                          mx.ndarray.one_hot(best_hyp_pos_indices, depth=source_length, dtype='int32'))

            #(7) update previous alignment
            if alignment_based:
                alignment = mx.nd.take(alignment,best_hyp_indices)
                alignment[:,t] = best_hyp_pos_indices
                alignment_jump = alignment_jump_offset + alignment[:,t] - (alignment[:,t-1]
                                                                           if t>1
                                                                           else mx.nd.zeros_like(alignment[:,t]))
                alignment_jump = mx.nd.expand_dims(data=alignment_jump,axis=1)
                last_aligned = mx.nd.take(last_aligned,best_hyp_indices)
                last_aligned[:] = mx.nd.where(best_hyp_pos_indices>=0, best_hyp_pos_indices,last_aligned)

            # (8) determine which hypotheses in the beam are now finished
            finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
            if mx.nd.sum(finished).asscalar() == self.batch_size * self.beam_size:  # all finished
                break

            # (9) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices,best_hyp_pos_idx_indices)

            #DEBUGGING
            logger.info("coverage vector[t=%d]: %s",t,
                    '[' + ' '.join([str(coverage_vector.asnumpy()[0][i]) for i in range(len(coverage_vector.asnumpy()[0]))]) + ']')



        return sequences, attentions, scores_accumulated[:,:,0], lengths, coverage_vector, alignment

    def topk(self, actual_source_length, alignment_based, reference, scores, t):
        active_positions = self._active_positions(actual_source_length, alignment_based, t)
        sliced_scores = scores[:, active_positions, :]
        if reference is not None and len(reference) > 0:
            if t == 1 and self.batch_size == 1:
                batch_select = 0
                word_select = mx.nd.array(reference[:, t - 1].astype("int32"), dtype="int32", ctx=self.context)
            else:
                batch_select = mx.nd.arange(0, self.batch_size*self.beam_size, ctx=self.context)
                word_select = mx.nd.array(mx.nd.repeat(reference[:, t - 1].astype("int32"), self.beam_size), ctx=self.context)

            sliced_scores = sliced_scores[batch_select, :, word_select].expand_dims(axis=-1)

        k = self._effective_beam_size(sliced_scores, t)

        offset = mx.nd.array(
            np.repeat(np.arange(0, self.batch_size * self.beam_size, self.beam_size), k),
            dtype='int32', ctx=self.context)

        (best_hyp_indices_mx, best_hyp_pos_indices_mx, best_word_indices_mx), \
        scores_accumulated_mx = utils.smallest_k_mx_batched(
            matrix=sliced_scores,
            k=k,
            batch_size=self.batch_size,
            offset=offset,
            only_first_row=t == 1)  #

        if k != self.beam_size:
            best_hyp_indices_mx = self._pad_with_first_value(
                array=best_hyp_indices_mx,
                batch_size=self.batch_size,
                padding_length=(self.beam_size -k))
            best_hyp_pos_indices_mx = self._pad_with_first_value(
                array=best_hyp_pos_indices_mx,
                batch_size=self.batch_size,
                padding_length=(self.beam_size - k))
            best_word_indices_mx = self._pad_with_first_value(
                array=best_word_indices_mx,
                batch_size=self.batch_size,
                padding_length=(self.beam_size - k))
            scores_accumulated_mx = self._pad_with_first_value(
                array=scores_accumulated_mx,
                batch_size=self.batch_size,
                padding_length=(self.beam_size - k))

        if reference is not None and len(reference) > 0:
            best_word_indices_mx = mx.nd.array(mx.nd.repeat(reference[:, t - 1].astype("int32"), self.beam_size),
                                      ctx=self.context)

        #assert(mx.nd.nansum(scores_accumulated_mx - scores[best_hyp_indices_mx, best_hyp_pos_indices_mx, best_word_indices_mx]) == 0)
        return best_hyp_indices_mx, best_hyp_pos_indices_mx, best_word_indices_mx, scores_accumulated_mx

    def _effective_beam_size(self, sliced_scores, t):
        sliced_score_shape = sliced_scores.reshape((-4, self.batch_size, -1, 0, 0)).shape
        k = (sliced_score_shape[-1] * sliced_score_shape[-2] * 1 if t == 1 else sliced_score_shape[-3])
        k = min(self.beam_size, k)
        return k

    def _pad_with_first_value(self, array, batch_size, padding_length):
        array = array.reshape(batch_size, -1)
        padding = array[:, 0].expand_dims(axis=1)
        padding = mx.nd.repeat(padding, repeats=padding_length, axis=1)
        array = mx.nd.concat(array, padding).reshape(-1)
        return array

    def _active_positions(self, actual_source_length, alignment_based, t):
        if alignment_based:
            active_positions = slice(0, min(
                min(C.MAX_JUMP, max(actual_source_length) - t) + min(C.MAX_JUMP, t - 1) + 1,
                max(actual_source_length)
            ))
        else:
            active_positions = slice(0, 1)
        return active_positions

    def _get_best_from_beam(self,
                            sequences: mx.nd.NDArray,
                            attention_lists: mx.nd.NDArray,
                            accumulated_scores: mx.nd.NDArray,
                            lengths: mx.nd.NDArray,
                            coverage_vector: mx.nd.NDArray,
                            alignment: mx.nd.NDArray,
                            source: List[List[str]]) -> List[Translation]:
        """
        Return the best (aka top) entry from the n-best list.

        :param sequences: Array of word ids. Shape: (batch_size * beam_size, bucket_key).
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param accumulated_scores: Array of length-normalized negative log-probs.
        :param lengths translation lengths
        :param coverage_vector final source coverage (batch_size*beam_size,encoded_source_length)
        :param alignment final alignment used to generate the hypotheses (batch_size*beam_size,max_output_length)
        :param source: batch of source sequenecs
        :return: Top sequence, top attention matrix, top accumulated score (length-normalized
                 negative log-probs) and length.
        """
        utils.check_condition(sequences.shape[0] == attention_lists.shape[0] \
                              == accumulated_scores.shape[0] == lengths.shape[0] \
                              == coverage_vector.shape[0], "Shape mismatch")
        # sequences & accumulated scores are in latest 'k-best order', thus 0th element is best
        best = 0
        result = []
        for sent in range(self.batch_size):
            idx = sent * self.beam_size + best
            length = int(lengths[idx].asscalar())
            sequence = sequences[idx][:length].asnumpy().tolist()
            # attention_matrix: (target_seq_len, source_seq_len)
            attention_matrix = np.stack(attention_lists[idx].asnumpy()[:length, :], axis=0)
            score = accumulated_scores[idx].asscalar()
            coverage = coverage_vector[idx].asnumpy()
            align = alignment[idx].asnumpy()
            result.append(Translation(sequence, attention_matrix, score, coverage, align, source[sent]))
            logger.info("max attention: %s. coverage vector: %s",
                        '[' + ' '.join([str(np.argmax(attention_matrix[j+1])) for j in range(length-1)]) + ']',
                        '[' + ' '.join([str(coverage[i]) for i in range(len(coverage))]) + ']')
        return result
