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
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric

import config
import constants as C

logger = logging.getLogger(__name__)


class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalization_type: How to normalize the loss.
    :param label_smoothing: Optional smoothing constant for label smoothing.
    """

    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalization_type: str,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)
        return [mx.sym.SoftmaxOutput(data=logits,
                                     label=labels,
                                     ignore_label=C.PAD_ID,
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name=C.SOFTMAX_NAME)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    def cross_entropy(self, pred, label, ignore):
        prob = mx.nd.pick(pred, label.astype(dtype="int32"))
        prob = prob * (1 - ignore) + ignore
        loss = -mx.nd.log(prob + 1e-8)  # pylint: disable=invalid-unary-operand-type
        return loss

    def cross_entropy_smoothed(self, pred, label, ignore):
        label_dist = mx.nd.one_hot(indices=label.astype(dtype='int32'),
                                   depth=self.loss_config.vocab_size,
                                   on_value=1.0 - self.loss_config.label_smoothing,
                                   off_value=self.loss_config.label_smoothing /
                                             (self.loss_config.vocab_size - 1.0))
        label_dist = mx.nd.where(1 - ignore, label_dist, mx.nd.zeros_like(label_dist))
        loss = label_dist * (- mx.nd.log(pred + 1e-8))  # pylint: disable=invalid-unary-operand-type
        return loss

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))
            # Ignore padding
            # TODO: contribute ignoring padding for cross-entropy back to MXNet
            ignore = (label == C.PAD_ID).astype(dtype=pred.dtype)

            if self.loss_config.label_smoothing > 0.0:
                loss = self.cross_entropy_smoothed(pred, label, ignore)
            else:
                loss = self.cross_entropy(pred, label, ignore)

            # Sum, normalizing if needed
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                loss = loss / mx.nd.sum(1 - ignore)
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size
            self.sum_metric += mx.nd.sum(loss).asscalar()