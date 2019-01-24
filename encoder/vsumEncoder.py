#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Simple Vsum Encoder (DAN Encoder)
Author: wangjia8@xiaomi.com
facebook: FastText (embbeding & linear layer has not imlemented here.) 
"""
import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from common.baseLayer import BaseLayer 


class VSumEncoder(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, emb):
        out = tf.reduce_sum(emb, 1)
        return out, int(out.get_shape()[1])

class WeightedSumEncoder(object):
    def __init__(self, input_x, **kwargs):
        self.input_x = input_x
        pass
    def __call__(self, emb):
        def mask(x):
            return tf.cast(tf.greater(tf.cast(x, tf.int32), tf.constant(0)), tf.float32)
        region_radius = int(int((self.input_x.get_shape()[1] - emb.get_shape()[1])) / 2)
        trimed_seq = self.input_x[..., region_radius:self.input_x.get_shape()[1] - region_radius]
        weight = tf.map_fn(mask, trimed_seq, dtype=tf.float32, back_prop=False)
        weight = tf.expand_dims(weight, -1)
        weighted_emb = emb * weight
        out = tf.reduce_sum(weighted_emb, 1)

        return out, int(out.get_shape()[1])
