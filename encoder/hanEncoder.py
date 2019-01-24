import sys, time
sys.path.append("../")
import tensorflow as tf
from attention.Attention import Attention
from tensorflow.contrib.rnn import BasicLSTMCell
from config.configParser import ConfigParser

"""
Brief: HAN Encoder
Author: wangjia8@xiaomi.com
reference:  Hierarchical Attention Networks for Document Classification
"""

class HANEncoder(object):
    def __init__(self, config, **kwargs):
        self.hidden_size = config.encoder_parameters.hidden_size

    def __call__(self, batch_embedding):
        rnn_outputs, _ = tf.nn.dynamic_rnn(BasicLSTMCell(self.hidden_size), batch_embedding, dtype=tf.float32)
        attention_res = Attention(rnn_outputs, self.hidden_size)
        return attention_res, attention_res.get_shape()[-1].value
