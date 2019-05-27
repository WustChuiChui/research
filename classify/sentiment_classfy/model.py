import sys, time
sys.path.append("../../")
import tensorflow as tf
from attention.Attention import attention
from common.baseLayer import linearLayer, fullConnectLayer  
from config.configParser import ConfigParser
from embedding.embeddingAdapter import EmbeddingAdapter
from loss.lossAdapter import LossAdapter
from encoder.encoderAdapter import EncoderAdapter

class Model(object):
    def __init__(self, config):
        #params
        self.max_len = config.model_parameters.max_len 
        self.learning_rate = config.model_parameters.learning_rate
        self.out_size = config.model_parameters.out_size

        #placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
        self.label = tf.placeholder(tf.int32, [None, None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")

    def buildGraph(self, config):
        #embedding
        batch_embedding = EmbeddingAdapter(config.model_parameters).getInstance()(self.x)

        #encode
        h_drop, hidden_size = EncoderAdapter(config=config, input_x=self.x).getInstance()(batch_embedding)

        #dense layer
        #y_hat = linearLayer(h_drop, hidden_size, self.out_size, need_highway=config.encoder_parameters.need_highway)
        y_hat = fullConnectLayer(h_drop, hidden_size, self.out_size)
        y_hat = tf.nn.dropout(y_hat, self.keep_prob)

        #prediction
        self.logits = tf.nn.softmax(y_hat, name="logits")
        self.prediction = tf.argmax(self.logits, 1, name="predict")
        
        #loss
        self.loss = LossAdapter(config.loss_parameters).getLoss(logits=y_hat, labels=self.label)

    def initOptimazer(self):
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,name='train_op')

if __name__ == '__main__':
    config = ConfigParser(config_file = "../../config/sentimentConfig")
    sentiment_model_obj = Model(config)
