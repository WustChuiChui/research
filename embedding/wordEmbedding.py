#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Word Embedding Layer
"""
import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer

class WordEmbedding(BaseLayer):
    """EmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, name="embedding", 
            initializer=None, **kwargs):
        BaseLayer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
                initializer=initializer)

    def _forward(self, seq):
        return tf.nn.embedding_lookup(self._W, seq)

class RegionAlignmentLayer(BaseLayer):
    def __init__(self, region_size, name="RegionAlig", **args):
        BaseLayer.__init__(self, name, **args)
        self._region_size = region_size

    def _forward(self, x): 
        region_radius = int(self._region_size / 2)
        aligned_seq = list(map(lambda i: tf.slice(x, [0, i - region_radius], [-1, self._region_size]), \
                range(region_radius, x.shape[1] - region_radius)))  #for python 3.x
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2]) 
        return aligned_seq

