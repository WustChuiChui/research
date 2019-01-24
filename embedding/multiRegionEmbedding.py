#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Multi Region Embedding
"""
import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding, RegionAlignmentLayer

class MultiRegionEmbedding(WordEmbedding):
    """"MultiRegionEmbedding"""
    def __init__(self, vocab_size, emb_size, region_sizes=[3,4,5], \
            region_merge_fn=tf.reduce_max, \
            name="multi_region_embedding", \
            initializer=None, \
            **kwargs):
        
        BaseLayer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_sizes = region_sizes[:]
        self._region_sizes.sort()
        self._region_merge_fn = region_merge_fn
        region_num = len(region_sizes)

        self._K = [None] * region_num
        self._K[-1] = tf.get_variable(name + '_K_%d' % (region_num - 1), \
                    shape=[vocab_size, self._region_sizes[-1], emb_size], \
                    initializer=initializer)

        for i in range(region_num - 1):
            st = int(self._region_sizes[-1]/2 - self._region_sizes[i]/2)
            ed = st + self._region_sizes[i]
            self._K[i] = self._K[-1][:, st:ed, :]

        super(MultiRegionEmbedding, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        """_forward
        """
        multi_region_emb = [] 

        for i, region_kernel in enumerate(self._K):
            region_radius = int(self._region_sizes[i] / 2)
            region_aligned_seq = RegionAlignmentLayer(self._region_sizes[i], name="RegionAlig_%d" % (i))(seq)
            region_aligned_emb = super(MultiRegionEmbedding, self)._forward(region_aligned_seq)
             
            trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
            context_unit = tf.nn.embedding_lookup(region_kernel, trimed_seq)

            projected_emb = region_aligned_emb * context_unit
            region_emb =  self._region_merge_fn(projected_emb, axis=2)
            multi_region_emb.append(region_emb)
        
        res_emb = multi_region_emb[0]
        for idx in range(1, len(multi_region_emb)):
            res_emb = tf.concat([res_emb, multi_region_emb[idx]], 1)
        return res_emb
