#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Charactor Embedding Layer
"""

import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding

class CharacterEmbedding(WordEmbedding):

    def __init__(self, vocab_size, emb_size, name="character_embedding",
        initializer=None, **kwargs):
        """
        Params: emb_size: embedding dim
                vocab_size:  character vocab size
                hiddden_size: hidden_layer size
                
        """
        BaseLayer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self.hidden_size = kwargs["hidden_size"] if "hidden_size" in kwargs else 64
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
            initializer=initializer) 

    def _forward(self, seq, character_len, **kwargs):
        """
        Params: seq should be batch_size * seq_lens * char_lens
		character_len should be batch_size * seq_len
        """
        character_embeddings = tf.nn.embedding_lookup(self._W, seq, name="character_embeddings")
        embeddings_shape = tf.shape(character_embeddings)

        character_embeddings = tf.reshape(character_embeddings, shape=[embeddings_shape[0]*embeddings_shape[1], embeddings_shape[-2], self._emb_size])
        character_lengths = tf.reshape(character_len, shape=[embeddings_shape[0]*embeddings_shape[1]])
               
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)

        _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, character_embeddings, sequence_length=character_lengths, dtype=tf.float32,time_major=False)
        _, ((_, output_hidden_state_fw), (_, output_hidden_state_bw)) = _output

        output = tf.concat([output_hidden_state_fw, output_hidden_state_bw], axis=-1)
        output = tf.reshape(output, shape=[embeddings_shape[0], embeddings_shape[1], 2 * self.hidden_size])
        return output
