import sys, time
sys.path.append("../")
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, BasicRNNCell
from common.indRNN import IndRNNCell
from attention.Attention import attention
from config.configParser import ConfigParser

"""
Brief: Rnn Encoder
"""

class RNNEncoder(object):
    def __init__(self, config, **kwargs):
        self.hidden_size = config.encoder_parameters.hidden_size
        self.keep_prob = config.encoder_parameters.keep_prob
        self.with_attention_layer = config.encoder_parameters.with_attention_layer if hasattr(config.encoder_parameters, "with_attention_layer") else False
        self.basic_cell = config.encoder_parameters.basic_cell if hasattr(config.encoder_parameters, "basic_cell") else "gru_cell"
        self.cell_dic = {"rnn_cell":BasicRNNCell,
                         "lstm_cell":BasicLSTMCell,
                         "gru_cell":GRUCell,
                         "indrnn_cell":IndRNNCell}

    def __call__(self, batch_embedding):
        if self.basic_cell in self.cell_dic:
            rnn_outputs, _ = bi_rnn(self.cell_dic[self.basic_cell](self.hidden_size),
                                    self.cell_dic[self.basic_cell](self.hidden_size),
                                    inputs=batch_embedding, dtype=tf.float32)
            print("Rnn encoder with " + self.basic_cell)
        else:
            rnn_outputs, _ = bi_rnn(GRUCell(self.hidden_size),
                                    GRUCell(self.hidden_size),
                                    inputs=batch_embedding, dtype=tf.float32)
            print("Rnn encoder with default GRU cell")
        if self.with_attention_layer:
            print("Build a Self-Attention Layer")
            return attention(rnn_outputs, self.keep_prob), self.hidden_size
        return tf.reduce_mean(rnn_outputs[0] + rnn_outputs[1], 1), self.hidden_size
