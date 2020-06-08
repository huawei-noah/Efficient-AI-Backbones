# 2020.02.26-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from the code: https://github.com/balancap/tf-imagenet/blob/master/models/mobilenet/mobilenet_v2.py
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import numpy as np

import tensorflow as tf

from tensorpack.models import (
    MaxPooling, GlobalAvgPooling, BatchNorm, Dropout, BNReLU, FullyConnected)
from tensorpack.tfutils import argscope
from tensorpack.models.common import layer_register
from tensorpack.utils.argtools import shape2d

from imagenet_utils import ImageNetModel
import utils
from myconv2d import MyConv2D as Conv2D
from myconv2d import BNNoReLU, SELayer
from myconv2d import GhostModule as MyConv

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

slim = tf.contrib.slim

# =========================================================================== #
# GhostNet model.
# =========================================================================== #
class GhostNet(ImageNetModel):
    """GhostNet model.
    """
    def __init__(self, num_classes=1000, dw_code=None, ratio_code=None, se=1, data_format='NHWC', 
                 width=1.0, depth=1.0, lr=0.2, weight_decay = 0.00004, dropout_keep_prob=0.8,
                 label_smoothing=0.0):
        self.scope = 'MobileNetV2'
        self.num_classes = num_classes
        self.dw_code = dw_code
        self.ratio_code = ratio_code
        self.se = se
        self.depth = depth
        self.depth_multiplier = width
        self.data_format = data_format
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_keep_prob = dropout_keep_prob
        self.label_smoothing = label_smoothing
        self.image_shape = 224

    def get_logits(self, inputs):
        sc = ghostnet_arg_scope(
            data_format=self.data_format,
            weight_decay=self.weight_decay,
            use_batch_norm=True,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=0.001,
            regularize_depthwise=False)
        with slim.arg_scope(sc):
            with argscope(Conv2D, 
                  kernel_initializer=kernel_initializer):
                with argscope([Conv2D, BatchNorm], data_format=self.data_format):
                    logits, end_points = ghost_net(
                        inputs,
                        dw_code=self.dw_code,
                        ratio_code=self.ratio_code,
                        se=self.se,
                        num_classes=self.num_classes,
                        dropout_keep_prob=self.dropout_keep_prob,
                        min_depth=8,
                        depth_multiplier=self.depth_multiplier,
                        depth=self.depth,
                        conv_defs=None,
                        prediction_fn=tf.contrib.layers.softmax,
                        spatial_squeeze=True,
                        reuse=None,
                        scope=self.scope,
                        global_pool=False)
                    return logits


# =========================================================================== #
# Functional definition.
# =========================================================================== #
# Conv and Bottleneck namedtuple define layers of the GhostNet architecture
# Conv defines 3x3 convolution layers
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor', 'se'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor', 'se'])

# _CONV_DEFS specifies the GhostNet body
_CONV_DEFS_0 = [
    Conv(kernel=[3, 3], stride=2, depth=16, factor=1, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1, se=0),

    Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=48/16, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=72/24, se=0),

    Bottleneck(kernel=[5, 5], stride=2, depth=40, factor=72/24, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=40, factor=120/40, se=1),

    Bottleneck(kernel=[3, 3], stride=2, depth=80, factor=240/40, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=200/80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184/80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184/80, se=0),

    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=480/80, se=1),
    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=672/112, se=1),
    Bottleneck(kernel=[5, 5], stride=2, depth=160, factor=672/112, se=1),

    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=1),

    Conv(kernel=[1, 1], stride=1, depth=960, factor=1, se=0),
    Conv(kernel=[1, 1], stride=1, depth=1280, factor=1, se=0)
]

@layer_register(log_shape=True)
def DepthConv(x, kernel_shape, padding='SAME', stride=1, data_format='NHWC',
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    if data_format=='NHWC':
        in_channel = in_shape[3]
        stride_shape = [1, stride, stride, 1]
    elif data_format=='NCHW':
        in_channel = in_shape[1]
        stride_shape = [1, 1, stride, stride]
    out_channel = in_channel
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = kernel_initializer
    kernel_shape = shape2d(kernel_shape) #[kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, data_format=data_format)
    return activation(conv, name='output')

    
def ghostnet_base(inputs,
                  final_endpoint=None,
                  min_depth=8,
                  depth_multiplier=1.0,
                  depth=1.0,
                  conv_defs=None,
                  output_stride=None,
                  dw_code=None,
                  ratio_code=None,
                  se=1,
                  scope=None):
    def depth(d):
        d = max(int(d * depth_multiplier), min_depth)
        d = round(d / 4) * 4
        return d
    
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS_0
        
    if dw_code is None or len(dw_code) < len(conv_defs):
        dw_code = [3] * len(conv_defs)
    print('dw_code', dw_code)
        
    if ratio_code is None or len(ratio_code) < len(conv_defs):
        ratio_code = [2] * len(conv_defs)
    print('ratio_code', ratio_code)
    
    se_code =  [x.se for x in conv_defs]
    print('se_code', se_code)
    
    if final_endpoint is None:
        final_endpoint = 'Conv2d_%d'%(len(conv_defs)-1)

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')
        
    with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1
            net = inputs
            in_depth = 3
            gi = 0
            for i, conv_def in enumerate(conv_defs):
                print('---')
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                    
                # change last bottleneck
                if i+2 == len(conv_defs):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, depth(conv_def.depth), [1, 1], stride=1, 
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    
                    ksize = utils.ksize_for_squeezing(net, 1024)
                    net = slim.avg_pool2d(net, ksize, padding='VALID',
                                          scope='AvgPool_7')
                    end_points[end_point] = net
                    
                # Normal conv2d.
                elif i+1 == len(conv_defs):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, 1280, conv_def.kernel, stride=conv_def.stride, 
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net
                    
                elif isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride, 
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net

                # Bottleneck block.
                elif isinstance(conv_def, Bottleneck):
                    # Stride > 1 or different depth: no residual part.
                    if layer_stride == 1 and in_depth == conv_def.depth:
                        res = net
                    else:
                        end_point = end_point_base + '_shortcut_dw'
                        res = DepthConv(end_point, net, conv_def.kernel, stride=layer_stride, 
                                        data_format='NHWC', activation=BNNoReLU)
                        end_point = end_point_base + '_shortcut_1x1'
                        res = Conv2D(end_point, res, depth(conv_def.depth), [1, 1], strides=1, data_format='NHWC',
                                     activation=BNNoReLU, use_bias=False)
                    
                    # Increase depth with 1x1 conv.
                    end_point = end_point_base + '_up_pointwise'
                    net = MyConv(end_point, net, depth(in_depth * conv_def.factor), [1, 1], dw_code[gi], ratio_code[gi], 
                                 strides=1, data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net
                    
                    # Depthwise conv2d.
                    if layer_stride > 1:
                        end_point = end_point_base + '_depthwise'
                        net = DepthConv(end_point, net, conv_def.kernel, stride=layer_stride, 
                                        data_format='NHWC', activation=BNNoReLU)
                        end_points[end_point] = net
                    # SE
                    if se_code[i] > 0 and se > 0:
                        end_point = end_point_base + '_se'
                        net = SELayer(end_point, net, depth(in_depth * conv_def.factor), 4)
                        end_points[end_point] = net
                        
                    # Downscale 1x1 conv.
                    end_point = end_point_base + '_down_pointwise'
                    net = MyConv(end_point, net, depth(conv_def.depth), [1, 1], dw_code[gi], ratio_code[gi], strides=1, 
                                 data_format='NHWC', activation=BNNoReLU if res is None else BNNoReLU, use_bias=False)
                    gi += 1
                        
                    # Residual connection?
                    end_point = end_point_base + '_residual'
                    net = tf.add(res, net, name=end_point) if res is not None else net
                    end_points[end_point] = net

                # Unknown...
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
                in_depth = conv_def.depth
                # Final end point?
                if final_endpoint in end_points:
                    return end_points[final_endpoint], end_points

    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def ghost_net(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 depth=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False,
                 dw_code=None,
                 ratio_code=None,
                 se=1,
                ):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = ghostnet_base(inputs, scope=scope, dw_code=dw_code, ratio_code=ratio_code,
                                                se=se, min_depth=min_depth, depth=depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1280
                net = Dropout('Dropout_1b', net, keep_prob=dropout_keep_prob)
                logits = Conv2D('Conv2d_1c_1x1', net, num_classes, 1, strides=1, 
                                 data_format='NHWC', activation=None)
                if spatial_squeeze:
                    logits = utils.spatial_squeeze(logits, scope='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


def ghostnet_arg_scope(is_training=True,
                           data_format='NHWC',
                           weight_decay=0.00004,
                           use_batch_norm=True,
                           batch_norm_decay=0.99,
                           batch_norm_epsilon=0.001,
                           regularize_depthwise=False):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'scale': True,
        'data_format': data_format,
        'is_training': is_training,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_initializer = kernel_initializer
    if regularize_depthwise:
        depthwise_regularizer = weights_regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_initializer,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer):
                    # Data format scope...
                    data_sc = utils.data_format_scope(data_format)
                    with slim.arg_scope(data_sc) as sc:
                        return sc
