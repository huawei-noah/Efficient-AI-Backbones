'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of Apache License, Version 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License, Version 2.0 License for more details.
'''
# implementation of Ghost module
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args
from tensorpack.models import BatchNorm, BNReLU, Conv2D
import numpy as np
import math
import utils
__all__ = ['GhostModule', 'SELayer', 'MyConv2D', 'BNNoReLU']

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

@layer_register(log_shape=True)
def MyDepthConv(x, kernel_shape, channel_mult=1, padding='SAME', stride=1, rate=1, data_format='NHWC',
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    if data_format=='NHWC':
        in_channel = in_shape[3]
        stride_shape = [1, stride, stride, 1]
    elif data_format=='NCHW':
        in_channel = in_shape[1]
        stride_shape = [1, 1, stride, stride]
    out_channel = in_channel * channel_mult

    if W_init is None:
        W_init = kernel_initializer
    kernel_shape = shape2d(kernel_shape) #[kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('DW', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, rate=[rate,rate], data_format=data_format)
    if activation is None:
        return conv
    else:
        return activation(conv, name='output')

    
def GhostModule(name, x, filters, kernel_size, dw_size, ratio, padding='SAME', strides=1, data_format='NHWC', use_bias=False,
                activation=tf.identity):
    with tf.variable_scope(name):
        init_channels = math.ceil(filters / ratio)
        x = Conv2D('conv1', x, init_channels, kernel_size, strides=strides, activation=activation, data_format=data_format,
                   kernel_initializer=kernel_initializer, use_bias=use_bias)
        if ratio == 1:
            return x #activation(x, name='output')
        dw1 = MyDepthConv('dw1', x, [dw_size,dw_size], channel_mult=ratio-1, stride=1, data_format=data_format, activation=activation)
        dw1 = dw1[:,:,:,:filters-init_channels] if data_format=='NHWC' else dw1[:,:filters-init_channels,:,:]
        x = tf.concat([x, dw1], 3 if data_format=='NHWC' else 1)
        return x


@layer_register(log_shape=True)
def SELayer(x, out_dim, ratio):
    squeeze = utils.spatial_mean(x, keep_dims=True, scope='global_pool')
    excitation = Conv2D('fc1', squeeze, int(out_dim / ratio), 1, strides=1, kernel_initializer=kernel_initializer, 
                             data_format='NHWC', activation=None)
    excitation = tf.nn.relu(excitation, name='relu')
    excitation = Conv2D('fc2', excitation, out_dim, 1, strides=1, kernel_initializer=kernel_initializer, 
                             data_format='NHWC', activation=None)
    excitation = tf.clip_by_value(excitation, 0, 1, name='hsigmoid')
    scale = x * excitation
    return scale


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def MyConv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=kernel_initializer,#tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        ):
    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret


@layer_register(use_scope=None)
def BNNoReLU(x, name=None):
    """
    A shorthand of BatchNormalization.
    """
    if name is None:
        x = BatchNorm('bn', x)
    else:
        x = BatchNorm(name, x)
    return x
