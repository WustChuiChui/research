import sys, time
sys.path.append("../../")
import tensorflow as tf
from common.baseLayer import linearLayer, fullConnectLayer  
from config.configParser import ConfigParser
from embedding.embeddingAdapter import EmbeddingAdapter
from loss.lossAdapter import LossAdapter
from encoder.encoderAdapter import EncoderAdapter

class Model(object):
    def __init__(self, config):
        self.tags_size = config.model_parameters.tag_size
        self.learning_rate = config.model_parameters.learning_rate 

        self.add_placeholders()

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name="input_x")
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
        self.keep_drop = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_lengths")

    def buildGraph(self, config):
        batch_embedding = EmbeddingAdapter(config.model_parameters).getInstance()(self.inputs)

        h_drop  = EncoderAdapter(config=config, keep_ori=True).getInstance()(batch_embedding) #(batch_size, seq_len, 2 * hidden_size)

        output = tf.reshape(h_drop, [-1, 2 * config.encoder_parameters.hidden_size])  #(batch_size * seq_len, 2 * hidden_size)
        #output = linearLayer(output, 2 * config.encoder_parameters.hidden_size, self.tags_size, self.keep_drop) #(batch_size * seq_len, tags_size) 
        output = fullConnectLayer(output, 2 * config.encoder_parameters.hidden_size, self.tags_size)    
        output = tf.nn.dropout(output, self.keep_drop)

        self.logits = tf.reshape(output, [-1, tf.shape(h_drop)[1], self.tags_size], name="logits") #(batch_size, seq_len, tags_size)
        
        self.loss = self.getCRFLikelihoodLoss(self.logits, self.tags, self.sequence_lengths)
        viterbi_seq, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.trans_params, self.sequence_lengths)
        self.viterbi_seq = tf.identity(viterbi_seq, name="viterbi_seq_out")

    def getCRFLikelihoodLoss(self, logits, tags, sequence_lengths):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, tags, sequence_lengths)
        self.trans_params = trans_params
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def initOptimazer(self):
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,name='train_op')

if __name__ == '__main__':
    config = ConfigParser(config_file = "../../config/ticketConfig")
    slot_filling_obj = Model(config)
