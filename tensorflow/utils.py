# ==============================================================================
# Copyright 2018 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Misc. collection of useful layers, mostly very simple!
"""
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorpack.callbacks.param import HyperParamSetter

slim = tf.contrib.slim

# =========================================================================== #
# Tools...
# =========================================================================== #

def _get_dimension(shape, dim, min_rank=1):
    """Returns the `dim` dimension of `shape`, while checking it has `min_rank`.
    Args:
        shape: A `TensorShape`.
        dim: Integer, which dimension to return.
        min_rank: Integer, minimum rank of shape.
    Returns:
        The value of the `dim` dimension.
    Raises:
        ValueError: if inputs don't have at least min_rank dimensions, or if the
            first dimension value is not defined.
    """
    dims = shape.dims
    if dims is None:
        raise ValueError('dims of shape must be known but is None')
    if len(dims) < min_rank:
        raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                        len(dims)))
    value = dims[dim].value
    if value is None:
        raise ValueError(
            'dimension %d of shape must be known but is None: %s' % (dim, shape))
    return value


# =========================================================================== #
# Extension of TensorFlow common layers.
# =========================================================================== #
@add_arg_scope
def channel_dimension(shape, data_format='NHWC', min_rank=1):
    """Returns the channel dimension of shape, while checking it has min_rank.
    Args:
        shape: A `TensorShape`.
        data_format: `NCHW` or `NHWC`.
        min_rank: Integer, minimum rank of shape.
    Returns:
         value of the first dimension.
    Raises:
        ValueError: if inputs don't have at least min_rank dimensions, or if the
            first dimension value is not defined.
    """
    return _get_dimension(shape, 1 if data_format == 'NCHW' else -1,
                          min_rank=min_rank)

@add_arg_scope
def channel_to_last(inputs, data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a consistent NHWC output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net

@add_arg_scope
def to_nhwc(inputs, data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a consistent NHWC output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'to_nhwc', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net

@add_arg_scope
def to_nchw(inputs, data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a consistent NHWC output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'to_nchw', [inputs]):
        if data_format == 'NHWC':
            net = tf.transpose(inputs, perm=(0, 3, 1, 2))
        elif data_format == 'NCHW':
            net = inputs
        return net

@add_arg_scope
def channel_to_hw(inputs, factors=[1, 1], data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a consistent NHWC output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'channel_to_hw', [inputs]):
        net = inputs
        if factors[0] == 1 and factors[1] == 1:
            return net

        if data_format == 'NCHW':
            net = tf.transpose(net, perm=(0, 2, 3, 1))
        # Inputs in NHWC format.
        shape = net.get_shape().as_list()
        shape[1] = int(shape[1] / factors[0])
        shape[2] = int(shape[2] / factors[1])
        shape[3] = -1
        net = tf.reshape(net, shape)
        # Original format.
        if data_format == 'NCHW':
            net = tf.transpose(net, perm=(0, 3, 1, 2))
        return net

@add_arg_scope
def concat_channels(l_inputs, data_format='NHWC', scope=None):
    """Concat a list of tensors on the channel axis.

    Args:
      inputs: List Tensors;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'concat_channels', l_inputs):
        if data_format == 'NHWC':
            net = tf.concat(l_inputs, axis=3)
        elif data_format == 'NCHW':
            net = tf.concat(l_inputs, axis=1)
        return net

@add_arg_scope
def split_channels(inputs, nsplits, data_format='NHWC', scope=None):
    """Split a tensor on the channel axis.

    Args:
      inputs: List Tensors;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'split_channels', [inputs]):
        if data_format == 'NHWC':
            nets = tf.split(inputs, nsplits, axis=3)
        elif data_format == 'NCHW':
            nets = tf.split(inputs, nsplits, axis=1)
        return nets

@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net

@add_arg_scope
def pad_logits(logits, pad=(0, 0)):
    """Pad logits Tensor, to deal with different number of classes.
    """
    shape = logits.get_shape().as_list()
    dtype = logits.dtype
    l = [logits]
    if pad[0] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[0]))
        l = [a] + l
    if pad[1] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[1]))
        l = l + [a]
    output = tf.concat(l, axis=1)
    return output

@add_arg_scope
def spatial_mean(inputs, scaling=None, keep_dims=False,
                 data_format='NHWC', scope=None):
    """Average tensor along spatial dimensions.

    Args:
      inputs: Input tensor;
      keep_dims: Keep spatial dimensions?
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_mean', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.reduce_mean(inputs, axes, keep_dims=keep_dims)
        return net

@add_arg_scope
def spatial_squeeze(inputs, data_format='NHWC', scope=None):
    """Squeeze spatial dimensions, if possible.

    Args:
      inputs: Input tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_squeeze', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.squeeze(inputs, axes)
        return net

@add_arg_scope
def ksize_for_squeezing(inputs, default_ksize=1024, data_format='NHWC'):
    """Get the correct kernel size for squeezing an input tensor.
    """
    shape = inputs.get_shape().as_list()
    kshape = shape[1:3] if data_format == 'NHWC' else shape[2:]
    if kshape[0] is None or kshape[1] is None:
        kernel_size_out = [default_ksize, default_ksize]
    else:
        kernel_size_out = [min(kshape[0], default_ksize),
                           min(kshape[1], default_ksize)]
    return kernel_size_out

@add_arg_scope
def batch_norm(inputs,
               activation_fn=None,
               normalizer_fn=None,
               normalizer_params=None):
    """Batch normalization layer compatible with the classic conv. API.
    Simpler to use with arg. scopes.
    """
    outputs = inputs
    # BN...
    if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs

@add_arg_scope
def drop_path(inputs, keep_prob, is_training=True, scope=None):
    """Drops out a whole example hiddenstate with the specified probability.
    """
    with tf.name_scope(scope, 'drop_path', [inputs]):
        net = inputs
        if is_training:
            batch_size = tf.shape(net)[0]
            noise_shape = [batch_size, 1, 1, 1]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)
            net = tf.div(net, keep_prob) * binary_tensor
        return net

# =========================================================================== #
# Useful methods
# =========================================================================== #
def data_format_scope(data_format):
    """Create the default scope for a given data format.
    Tries to combine all existing layers in one place!
    """
    with slim.arg_scope([slim.conv2d,
                         slim.separable_conv2d,
                         slim.max_pool2d,
                         slim.avg_pool2d,
                         slim.batch_norm,
                         concat_channels,
                         split_channels,
                         channel_to_last,
                         to_nchw,
                         to_nhwc,
                         channel_to_hw,
                         spatial_squeeze,
                         spatial_mean,
                         ksize_for_squeezing,
                         channel_dimension],
                        data_format=data_format) as sc:
        return sc

class HyperParamSetterWithCosine(HyperParamSetter):
    """ Set the parameter by a function of epoch num. """
    def __init__(self, param, base_lr, start_step, n_step, step_based=True):
        """
        Cosine learning rate
        """
        super(HyperParamSetterWithCosine, self).__init__(param)
        self._base_lr = base_lr
        self._start_step = start_step
        self._n_step = n_step
        self._step = step_based

    def _get_value_to_set(self):
        refnum = self.global_step if self._step else self.epoch_num
        if self._start_step > refnum:
            return None
        return 0.5*self._base_lr*(1+np.cos(np.pi*(refnum-self._start_step)/self._n_step))

    def _trigger_epoch(self):
        if not self._step:
            self.trigger()

    def _trigger_step(self):
        if self._step:
            self.trigger()