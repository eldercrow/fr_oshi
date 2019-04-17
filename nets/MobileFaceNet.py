# Copyright 2018 The AI boy xsr-ai. All Rights Reserved.
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
"""MobileFaceNets.

MobileFaceNets, which use less than 1 million parameters and are specifically tailored for high-accuracy real-time
face verification on mobile and embedded devices.

here is MobileFaceNets architecture, reference from MobileNet_V2 (https://github.com/xsr-ai/MobileNetv2_TF).

As described in https://arxiv.org/abs/1804.07573.

  MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices

  Sheng Chen, Yang Liu, Xiang Gao, Zhen Han

"""
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from contextlib import contextmanager

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_variable_scope
from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.models import (
    Conv2D, FullyConnected, MaxPooling, BatchNorm, BNReLU, PReLU, LinearWrap)
from tensorpack import layer_register


global_conv_counter = 0
def _get_conv_name():
    if global_conv_counter == 0:
        pref = 'Conv'
    else:
        pref = 'Conv_{}'.format(global_conv_counter)
    global_conv_counter += 1
    return pref


global_dwconv_counter = 0
def _get_dwconv_name():
    if global_dwconv_counter == 0:
        pref = 'SeparableConv2d'
    else:
        pref = 'SeparableConv2d_{}'.format(global_dwconv_counter)
    global_dwconv_counter += 1
    return pref


@layer_register(use_scope=None)
def BNonly(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.
    """
    x = BatchNorm('bn', x)
return x


@layer_register(use_scope=None)
def BNPReLU(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.
    """
    x = BatchNorm('bn', x)
    x = PReLU('prelu', x)
return x


@layer_register(log_shape=True)
def DWConv(x, kernel, padding='SAME', stride=1, w_init=None, activation=BNPReLU, dilate=1, data_format='NHWC'):
    '''
    Depthwise conv + BN + (optional) ReLU.
    We do not use channel multiplier here (fixed as 1).
    '''
    assert data_format in ('NHWC', 'channels_last')
    channel = x.get_shape().as_list()[-1]
    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel, kernel]
    if not isinstance(dilate, (list, tuple)):
        dilate = [dilate, dilate]
    filter_shape = [kernel[0], kernel[1], channel, 1]

    if w_init is None:
        w_init = tf.variance_scaling_initializer(2.0)
    W = tf.get_variable('W', filter_shape, initializer=w_init)
    out = tf.nn.depthwise_conv2d(x, W,
                                 [1, stride, stride, 1],
                                 padding=padding,
                                 rate=dilate,
                                 data_format=data_format)

    if activation is None:
        return out
    return activation(out)


# @layer_register(log_shape=True)
def LinearBottleneck(x, ich, och, kernel,
                     padding='SAME',
                     stride=1,
                     activation=BNPReLU,
                     t=3,
                     w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    if active is None:
        active = True if kernel > 3 else False

    out = Conv2D(_get_conv_name(), x, int(ich*t), 1, activation=BNPReLU)
    out = DWConv(_get_dwconv_name(), out, kernel, padding, stride, w_init, activation=activation)
    out = Conv2D(_get_conv_name(), out, och, 1, activation=BNonly)
    if stride != 1:
        return out
    if ich != och:
        x = Conv2D(_get_conv_name(), x, int(och), 1, activation=BNonly)
    return x + out


def mobilenetv2_base(inputs):
    '''
    '''
    l = Conv2D('Conv2D_0', inputs, 64, 3, strides=2, activation=BNPReLU)
    l = DWConv(_get_dwconv_gname(), l, 3, activation=BNPReLU)
    l = Conv2D(_get_conv_name(), l, 64, 1, activation=BNonly)

    for ii in range(5):
        stride = 2 if ii = 0 else 1
        l = LinearBottleneck(l, 64, 64, 3, stride=stride, t=2)
    l = LinearBottleneck(l, 64, 128, 3, stride=2, t=4)
    for ii in range(6):
        l = LinearBottleneck(l, 128, 128, 3, t=2)
    l = LinearBottleneck(l, 128, 128, 3, stride=2, t=4)
    for ii in range(2):
        l = LinearBottleneck(l, 128, 128, 3, t=2)

    l = Conv2D(_get_conv_name(), l, 512, 1, activation=BNPReLU)
    return l
    # hard coded image size
    # assert l.get_shape.as_list()[1] == 7
    # assert l.get_shape.as_list()[2] == 7


def mobilenetv2(inputs):
    with tf.variable_scope('MobileFaceNet'):
        l = mobilenetv2_base(inputs)

        with tf.variable_scope('Logits'):
            l = DWConv('SeparableConv2d', l, 7, activation=BNonly, padding='VALID')
            l = Conv2D('Conv', l, 512, 1, activation=BNonly)

            bottleneck_layer_size = 128

            l = Conv2D('LinearConv1x1', l, bottleneck_layer_size, 1, activation=BNonly)
            l = tf.squeeze(l, [1, 2], name='SpatialSqueeze')
    return l
