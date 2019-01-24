#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
ContextWordEmbedding
"""
import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding, RegionAlignmentLayer

class ContextWordRegionEmbedding(WordEmbedding):
    """ContextWordRegionEmbedding"""
    def __init__(self, vocab_size, emb_size, region_size=3, 
            region_merge_fn=tf.reduce_max,
            name="embedding",
            initializer=None, **kwargs):
        super(ContextWordRegionEmbedding, self).__init__(vocab_size * region_size, emb_size, name,
                initializer, **kwargs)
        self._region_merge_fn = region_merge_fn
        self._word_emb = tf.get_variable(name + '_wordmeb', shape=[vocab_size, emb_size], 
                initializer=initializer)
        self._unit_id_bias = np.array([i * vocab_size for i in range(region_size)])
        self._region_size = region_size

    def _region_aligned_units(self, seq):
        """
        _region_aligned_unit
        """
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_seq = region_aligned_seq + self._unit_id_bias
        region_aligned_unit = super(ContextWordRegionEmbedding, self)._forward(region_aligned_seq)
        return region_aligned_unit
    
    def _forward(self, seq):
        """forward
        """
        region_radius = int(self._region_size / 2)
        word_emb = tf.nn.embedding_lookup(self._word_emb, \
                tf.slice(seq, \
                [0, region_radius], \
                [-1, tf.cast(seq.get_shape()[1] - 2 * region_radius, tf.int32)]))
        word_emb = tf.expand_dims(word_emb, 2)
        region_aligned_unit = self._region_aligned_units(seq)
        embedding = region_aligned_unit * word_emb
        embedding = self._region_merge_fn(embedding, axis=2)
        return embedding
