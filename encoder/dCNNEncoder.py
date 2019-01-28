import sys,time
sys.path.append("../")
import tensorflow as tf
from config.configParser import ConfigParser
from common.baseLayer import fullConnectLayer
from common.activations import ActivationAdapter

"""
Brief: DCNN Encoder
refenrence: A Convolutional Neural Network for Modelling Sentences
"""

class DCNNEncoder(object):
    def __init__(self, config, **kwargs):
        self.batch_size = config.model_parameters.batch_size
        self.max_len = config.model_parameters.max_len
        self.num_filters = [6, 14]
        self.embedding_size = config.model_parameters.embedding_size
        self.top_k = 3
        self.k1 = 11
        self.out_size = config.model_parameters.out_size
        self.ws = [7, 5]
        self.act_func = ActivationAdapter(config).getInstance()

    def per_dim_conv(self, x, w, b):
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)

        convs = []
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                conv = self.act_func(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
        return conv

    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i + 1])
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)
        return fold

    def __call__(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        W1 = tf.Variable(tf.truncated_normal([self.ws[0], self.embedding_size, 1, self.num_filters[0]], stddev=0.01), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[0], self.embedding_size]), name="b1")
        conv1 = self.per_dim_conv(inputs, W1, b1)
        conv1 = self.fold_k_max_pooling(conv1, self.k1)

        W2 = tf.Variable(tf.truncated_normal([self.ws[1], int(self.embedding_size/2), self.num_filters[0], self.num_filters[1]], stddev=0.01), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[1], self.embedding_size]), name="b2")
        conv2 = self.per_dim_conv(conv1, W2, b2)
        
        fold = self.fold_k_max_pooling(conv2, self.top_k)
        hidden_size = int(self.top_k * self.embedding_size * self.num_filters[1] / 4)
        fold_flatten = tf.reshape(fold, [-1, hidden_size])
        return fold_flatten, hidden_size
