import sys, time
sys.path.append("../")
import tensorflow as tf
from config.configParser import ConfigParser

"""
Brief: IDCNN Encoder
Author: wangjia8@xiaomi.com
refenrence: End to End Chinese Named Entity Recognition by Iterated Dilated Convolution Nerual Networks with Conditional Random Field layer.
"""

class IDCNNEncoder(object):
    def __init__(self, config, **kwargs):
        self.layers=[1, 1, 2]
        self.filter_width = 3
        self.embedding_size = config.model_parameters.embedding_size
        self.num_filter = config.encoder_parameters.num_filters
        self.repeat_times = 4

    def __call__(self, inputs):
        inputs = tf.expand_dims(inputs, 1)
        shape = [1, self.filter_width, self.embedding_size, self.num_filter]
        filter_weights = tf.Variable(tf.truncated_normal(shape=[1, self.filter_width, self.embedding_size, self.num_filter], stddev=0.1), name="filters_weights")

        layer_input = tf.nn.conv2d(inputs, filter_weights, strides=[1,1,1,1], padding="SAME")
        final_layers = []
        total_last_dim = 0
        for j in range(self.repeat_times):
            for i in range(len(self.layers)):
                dilation = self.layers[i]
                isLast = True if i == (len(self.layers) - 1) else False
                with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=tf.AUTO_REUSE):
                    w = tf.Variable(tf.truncated_normal(shape=[1, self.filter_width, self.num_filter, self.num_filter], stddev=0.1))
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filter]), name="b")
                    conv = tf.nn.atrous_conv2d(layer_input, w, rate=dilation, padding="SAME")
                    conv = tf.nn.relu(tf.nn.bias_add(conv, b))
                    if isLast:
                        final_layers.append(conv)
                        total_last_dim += self.num_filter
                    layer_input = conv
        final_out = tf.concat(axis=3, values=final_layers)
        final_out = tf.squeeze(final_out, [1])
        hidden_size = final_out.get_shape()[1].value * total_last_dim
        final_out = tf.reshape(final_out, [-1, hidden_size])
        return final_out, hidden_size
