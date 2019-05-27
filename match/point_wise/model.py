import sys, time
sys.path.append("../../")
import tensorflow as tf
from attention.Attention import attention
from common.baseLayer import linearLayer  
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
        self.query_1 = tf.placeholder(tf.int32, [None, self.max_len])
        self.query_2 = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32)

    def buildGraph(self, config):
        #embedding
        embedding = EmbeddingAdapter(config.model_parameters).getInstance()
        batch_embedding_1 = embedding(self.query_1)
        batch_embedding_2 = embedding(self.query_2)

        #encode
        encoder = EncoderAdapter(config=config, input_x=self.query_1).getInstance()
        h_drop_1, hidden_size = encoder(batch_embedding_1)
        h_drop_2, _ = encoder(batch_embedding_2)

        #dense layer
        h_drop = tf.concat([h_drop_1, h_drop_2], axis=-1)
        y_hat = linearLayer(h_drop, hidden_size * 2, self.out_size, self.keep_prob, need_highway=config.encoder_parameters.need_highway)

        #prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
        
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
    config = ConfigParser(config_file = "../../config/atecMatchConfig")
    sentiment_model_obj = Model(config)
