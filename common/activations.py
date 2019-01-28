import logging
import tensorflow as tf

class ActivationAdapter(object):
    def __init__(self, config, **kwargs):
        self.activation_type = config.encoder_parameters.activation_type \
            if hasattr(config.encoder_parameters, "activation_type") else "swish"
        self.actication_dicts = {"relu":tf.nn.relu,
                                 "sigmoid":tf.nn.sigmoid,
                                 "tanh":tf.nn.tanh,
                                 "leaky_relu":self.leakyRelu,
                                 "elu":tf.nn.elu,
                                 "selu":tf.nn.selu,  #f(x) = lamda * x (x > 0) else lamda * (alpha * exp(x) - alpha),
                                 "swish":self.swish,
                                 "sin":tf.sin,
                                 "cube":self.cube,
                                 "penalized_tanh":self.penalized_tanh,
                                 "cosper":self.cosper,
                                 "minsin":self.minsin,
                                 "tanhrev":self.tanhrev,
                                 "maxsig":self.maxsig,
                                 "maxtanh":self.maxtanh,
                                 "softplus":tf.nn.softplus,
                                 "softsign":tf.nn.softsign,
                                 "linear":self.linear
                                 }

    def linear(self, x, name="linear"):
        with tf.variable_scope(name):
            return x

    def softplus(self, x, name="softplus"):
        with tf.variable_scope(name):
            return tf.nn.softplus

    def swish(self, x, name="swish"):
        """
        @brief: f(x) = sigmoid(x) * x
        """
        with tf.variable_scope(name):
            return (tf.nn.sigmoid(x * 1.0) * x)

    def leakyRelu(self, x, leak=0.2, name="leaky_relu"):
        """
        @brief: f(x) = max(alpha * x, x), solved dead ReLU problem
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

    def cube(self, x, name="cube_act"):
        """
        @brief: f(x) = pow(x, 3)
        """
        with tf.variable_scope(name):
            return tf.pow(x, 3)

    def penalized_tanh(self, x, name="penalized_tanh"):
        """
        @brief: f(x) = max(tanh(x), alpha * tanh(x))
        """
        with tf.variable_scope(name):
            alpha = 0.25
            return tf.maximum(tf.tanh(x), alpha * tf.tanh(x))

    def cosper(self, x, name="cosper_act"):
        """
        @brief: f(x) = cos(x) - x
        """
        with tf.variable_scope(name):
            return (tf.cos(x) - x)

    def minsin(self, x, name="minsin_act"):
        """
        @brief: f(x) = min(x, xin(x))
        """
        with tf.variable_scope(name):
            return tf.minimum(x, tf.sin(x))

    def tanhrev(self, x, name="tanhprev"):
        """
        @brief: f(x) = pow(atan(x), 2) - x
        """
        with tf.variable_scope(name):
            return (tf.pow(tf.atan(x), 2) - x)

    def maxsig(self, x, name="maxsig_act"):
        """
        @brief: f(x) = max(x, tf.sigmiod(x))
        """
        with tf.variable_scope(name):
            return tf.maximum(x, tf.sigmoid(x))

    def maxtanh(self, x, name="max_tanh_act"):
        """
        @brief: f(x) = max(x, tanh(x))
        """
        with tf.variable_scope(name):
            return tf.maximum(x, tf.tanh(x))

    def getInstance(self):
        if self.activation_type in self.actication_dicts:
            logging.info("activation use %s" % (self.activation_type))
            return self.actication_dicts[self.activation_type]
        else:
            logging.WARN("activation param is invalid. ")
            return self.swish
