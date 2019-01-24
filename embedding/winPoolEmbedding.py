#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Win Pooling Embedding
"""
import sys
import tensorflow as tf
import numpy as np
sys.path.append("../")
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding, RegionAlignmentLayer

class WinPoolEmbedding(WordEmbedding):
    """WindowPoolEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size=3, \
            region_merge_fn=tf.reduce_max, \
            name="win_pool_embedding", \
            initializer=None, \
            **kwargs):
        
        BaseLayer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, emb_size],
                initializer=initializer)
        super(WinPoolEmbedding, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WinPoolEmbedding, self)._forward(region_aligned_seq)

        return self._region_merge_fn(region_aligned_emb, axis=2)

