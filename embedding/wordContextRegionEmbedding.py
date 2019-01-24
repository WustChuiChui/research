#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
WordContextRegionEmbedding
"""
import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding, RegionAlignmentLayer

class WordContextRegionEmbedding(WordEmbedding):
    """WordContextRegionEmbedding"""
    def __init__(self, vocab_size, emb_size, region_size=3, \
            region_merge_fn=tf.reduce_max, \
            name="word_context_region_embedding", \
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
        super(WordContextRegionEmbedding, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WordContextRegionEmbedding, self)._forward(region_aligned_seq)

        region_radius = int(self._region_size / 2)
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)
