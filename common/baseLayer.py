#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import tensorflow as tf
from functools import wraps


class BaseLayer(object):
    """Layer"""
    def __init__(self, name, activation=None, dropout=None, decay_mult=None):
        self._name = name
        self._activation = activation
        self._dropout = dropout
        self._decay_mult = decay_mult

    def get_variable(self, name, **kwargs):
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult
        return tf.get_variable(name, **kwargs)

    def __call__(self, *inputs):
        outputs = []
        for x in inputs:
            if type(x) == tuple or type(x) == list:
                y = self._forward(*x)
            else:
                y = self._forward(x)
            if self._activation:
                y = self._activation(y)
            if self._dropout:
                if hasattr(tf.flags.FLAGS, 'training'):
                    y = tf.cond(tf.flags.FLAGS.training, 
                            lambda: tf.nn.dropout(y, keep_prob = 1.0 - self._dropout), 
                            lambda: y)
                else:
                    y = tf.nn.dropout(y, keep_prob = 1.0 - self._dropout)
            outputs.append(y)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def _forward(self, x):
        return x

def fullConnectLayer(inputs, in_size, out_size, keep_dropout = 1.0):
    W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[out_size]))
    y_hat = tf.nn.xw_plus_b(inputs, W, b) 
    #y_hat = tf.nn.dropout(y_hat, keep_dropout)
    return y_hat

def highWay(inputs, size, num_layers=2, fc=tf.nn.relu, scope="Highway"):
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = fc(fullConnectLayer(inputs, size, size))
            t = tf.sigmoid(fullConnectLayer(inputs, size, size))
            output = t * g + (1.0 - t) * inputs
            inputs = output
            print("Configure %s highway" % (idx + 1))
        return output

def linearLayer(inputs, in_size, out_size, keep_dropout=1.0, need_highway=False):
    if need_highway:
        inputs = highWay(inputs, in_size)
    return fullConnectLayer(inputs, in_size, out_size, keep_dropout)

def layerNormalization(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs
