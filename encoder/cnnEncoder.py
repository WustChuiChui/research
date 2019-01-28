#!/usr/bin/env python
#-*- coding: utf-8 -*- 

"""
TextCNN 
kim: CNN for text classfication
"""
import sys
sys.path.append("../")
import numpy as np
from common.activations import ActivationAdapter
import tensorflow as tf

class CNNEncoder(object):
    def __init__(self, config, **kwargs):
        self.filters_size = [3,4,5]
        self.embedding_size = config.model_parameters.embedding_size
        self.num_filters = config.encoder_parameters.num_filters
        self.seq_len = config.model_parameters.max_len
        self.act_func = ActivationAdapter(config).getInstance()

    def __call__(self, inputs):
        pooled_outputs = []
        inputs = tf.expand_dims(inputs, -1)
        for i, filter_size in enumerate(self.filters_size):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv = tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding="VALID", name="conv")
            h = self.act_func(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, self.seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filters_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool = tf.reshape(self.h_pool, [-1, num_filters_total])
        return self.h_pool, num_filters_total