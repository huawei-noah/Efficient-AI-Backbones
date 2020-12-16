# 2020.02.26-Changed for main script for testing GhostNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/imagenet-resnet.py
import argparse
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from time import time
import zipfile

import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger

from imagenet_utils import (
    get_imagenet_dataflow,
    ImageNetModel, GoogleNetResize, eval_on_ILSVRC12)


def get_data(name, batch):
    isTrain = name == 'train'
    image_shape = 224

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            GoogleNetResize(crop_area_fraction=0.49 if args.width_ratio < 1 else 0.08,
                           target_shape=image_shape),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                ]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(int(image_shape*256/224), cv2.INTER_CUBIC),
            imgaug.CenterCrop((image_shape, image_shape)),
        ]
    return get_imagenet_dataflow(args.data_dir, name, batch, augmentors, 
                       meta_dir = args.meta_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--data_dir', help='dataset dir.', type=str, default='/cache/data/imagenet/')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch', type=int, default=1024, help='total batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='base learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='total epochs')
    parser.add_argument('--load', help='path to load a model from', default='./ghostnet_chechpoint')
    parser.add_argument('--flops', type=int, help='print flops and exit', default=0)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.00004)
    parser.add_argument('--label_smoothing', type=float, help='label_smoothing', default=0.1)
    parser.add_argument('--data-format', help='image data format',
                        default='NHWC', choices=['NCHW', 'NHWC'])
    # param parser
    parser.add_argument('--width_ratio', help='width_ratio', type=float, default=1)
    parser.add_argument('--dropout_keep_prob', help='dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--se', help='se', type=int, default=3)
    parser.add_argument('--dw_code_str', help='dw_code_str', type=str, default='')
    parser.add_argument('--ratio_code_str', help='ratio_code_str', type=str, default='')
    args, unparsed = parser.parse_known_args()
    args.meta_dir = os.path.join(args.data_dir, 'caffe_ilsvrc12')
    print(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.batch != 1024:
        logger.warn("Total batch size != 1024, you need to change other hyperparameters to get the same results.")
    TOTAL_BATCH_SIZE = args.batch
    
    if len(args.dw_code_str) == 0:
        dw_code = None
    else:
        dw_code = [int(s) for s in args.dw_code_str.split(',')]
    print('dw_code', dw_code)
    
    if len(args.ratio_code_str) == 0:
        ratio_code = None
    else:
        ratio_code = [int(s) for s in args.ratio_code_str.split(',')]
    print('ratio_code', ratio_code)

    # create GhostNet
    from ghostnet import GhostNet
    model = GhostNet(width=args.width_ratio, se=args.se, 
                      weight_decay=args.weight_decay,
                      dw_code=dw_code, ratio_code=ratio_code,
                      label_smoothing=args.label_smoothing)
    model.data_format = args.data_format
    print('model created')

    # start evaluation
    if args.eval:
        batch = 256    # something that can run on your gpu
        ds = get_data('val', batch)
        start = time()
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
        stop = time()
        print('Evaluation used time: %.2fs.' % (stop-start))
    elif args.flops > 0:
        # manually build the graph with batch=1
        image_shape = 224
        input_desc = [
            InputDesc(tf.float32, [1, image_shape, image_shape, 3], 'input'),
            InputDesc(tf.int32, [1], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=False):
            model.build_graph(*input.get_input_tensors())
        model_utils.describe_trainable_vars()

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        logger.info("Note that TensorFlow counts flops in a different way from the paper.")
        logger.info("TensorFlow counts multiply+add as two flops, however the paper counts them "
                    "as 1 flop because it can be executed in one instruction.")
    else:
        print('nothing done')
        