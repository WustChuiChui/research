import sys, time
sys.path.append("../")
import tensorflow as tf
from attention.Attention import multiheadAttention, feedforward
from config.configParser import ConfigParser

"""
Brief: all Attention Encoder
Author: wangjia8@xiaomi.com
reference:  Attention Is All You Need
"""

class AttentionEncoder(object):
    def __init__(self, config, **kwargs):
        self.hidden_size = config.encoder_parameters.hidden_size
        self.embedding_size = config.model_parameters.embedding_size
        self.max_len = config.model_parameters.max_len

    def __call__(self, batch_embedding):
        mul_att = multiheadAttention(queries=batch_embedding, keys=batch_embedding)
        outputs = feedforward(mul_att, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        return outputs, outputs.get_shape()[-1].value
