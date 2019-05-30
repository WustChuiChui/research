#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
EnhancedWordEmbedding
"""

import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding


class EnhancedWordEmbedding(WordEmbedding):
    """EnhancedWordEmbedding"""
    def __init__(self, vocab_size, emb_size, name="enhancedEmbedding",
                initializer=None, zero_pad=True, scale=0.5, **kwargs):
        super(EnhancedWordEmbedding, self).__init__(vocab_size, emb_size, name, initializer)
        if zero_pad:
            self._W = tf.concat((tf.zeros(shape=[1, emb_size]), self._W[1:]), 0)
        self.scale = scale
        self.emb_size = emb_size


    def _forward(self, seq, **kwargs):
        outputs = super(EnhancedWordEmbedding, self)._forward(seq)
        
        outputs = outputs * (self.emb_size ** self.scale)
        return outputs
