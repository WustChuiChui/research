import sys, time
sys.path.append("../")
import tensorflow as tf
from config.configParser import ConfigParser

"""
Brief: DPCNN Encoder
Author: wangjia8@xiaomi.com
Tencent: Deep Pymirid Convolutional neural networks for text categorization
"""

class DPCNNEncoder(object):
    def __init__(self, config, **kwargs):
        #self.filter_sizes = config.encoder_parameters.filter_sizes
        self.num_filters = config.encoder_parameters.num_filters
        self.out_size = config.model_parameters.out_size

    def conv(self, x, filter_height, filter_width, filter_input, filter_output):
        filter_shape = [filter_height, filter_width, filter_input, filter_output]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[filter_output]), name="b")
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    def act_fun(self, inputs, name="act_fun"):
        with tf.name_scope(name):
            return tf.nn.relu(inputs, name="relu")

    def max_pool(self, inputs, k_height, k_width, stride):
        return tf.nn.max_pool(inputs, ksize=[1, k_height, k_width, 1], strides=[1, stride, 1, 1], padding="SAME")

    def block(self, inputs):
        pool_res = self.max_pool(inputs, k_height=2, k_width=1, stride=2)
        conv_res = self.conv(pool_res, filter_height=2, filter_width=1, filter_input=self.num_filters, filter_output=self.num_filters)
        act_fun_res = self.act_fun(conv_res)
        conv_res = self.conv(act_fun_res, filter_height=2, filter_width=1, filter_input=self.num_filters, filter_output=self.num_filters)
        block_res = tf.add(pool_res,conv_res)
        return block_res

    def forward(self, inputs):
        conv_res = self.conv(inputs, filter_height=1, filter_width=self.filter_out_size, filter_input=1, filter_output=self.num_filters) 
        act_fun_res = self.act_fun(conv_res)
        conv_res = self.conv(act_fun_res, filter_height=2, filter_width=1, filter_input=self.num_filters, filter_output=self.num_filters)
        return conv_res 

    def __call__(self, inputs):
        self.filter_out_size = inputs.get_shape()[-1].value
        embedded_expanded = tf.expand_dims(inputs, -1)
        output = self.forward(embedded_expanded)
        block_num = 1
        while output.get_shape()[1].value > 2 * self.out_size:
            output = self.block(output)
            block_num += 1
        output = self.max_pool(output, k_height=output.get_shape()[1].value, k_width=1, stride=1)
        output = tf.reduce_mean(output, -1)
        output_shape = output.get_shape()
        out_size = output_shape[1].value * output_shape[-1].value
        output = tf.reshape(output, [-1, out_size])
        return output, out_size
