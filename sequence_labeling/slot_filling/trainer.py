import sys, time, os
sys.path.append("../../")
from sequence_labeling.slot_filling.model import Model
from config.configParser import ConfigParser
from preprocess.preProcess import PreProcess
from preprocess.ioUtils import *
from preprocess.tfUtils import *

class Trainer(object):
    def __init__(self, config):
        print(config)
        self.data_dic = PreProcess()(config)
        self.config = config
        self.model = Model(config)
        print("buildGraph")
        self.model.buildGraph(config)
        print("buildGraph suceccessful")
        self.model.initOptimazer()
        self.data_dic = PreProcess()(config)

    def train_step(self, model, sess, batch):
        feed_dict = {model.inputs:batch[0],
                     model.tags:batch[1],
                     model.keep_drop:config.encoder_parameters.keep_prob,
                     model.sequence_lengths:np.array([config.model_parameters.max_len] * config.model_parameters.batch_size)}
        to_return = {"train_op":model.train_op,
                     "loss":model.loss,
                     "global_step":model.global_step}
        return sess.run(to_return, feed_dict)

    def eval_step(self, model, sess, batch):
        feed_dict = {model.inputs:batch[0],                                                       
                     model.tags:batch[1],                                                         
                     model.keep_drop:1,                         
                     model.sequence_lengths:np.array([config.model_parameters.max_len] * len(batch[0]))}
        viterbi_seq, viterbi_score = sess.run([model.viterbi_seq, model.viterbi_score], feed_dict=feed_dict)
        return viterbi_seq, viterbi_score

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for e in range(config.model_parameters.train_epoch):
            t0 = time.time()
            print("Epoch %d start !" % (e + 1)) 
            for x_batch, y_batch in fill_feed_dict(self.data_dic["train_data_x"], 
                                                  self.data_dic["train_data_tags"], 
                                                  config.model_parameters.batch_size,
                                                  need_shuffle=True):
                
                return_dict = self.train_step(self.model, sess, (x_batch, y_batch))

            t1 = time.time()
            print("dev")
            viterbi_seq, viterbi_score = self.eval_step(self.model, sess, (self.data_dic["dev_data_x"], self.data_dic["dev_data_tags"]))
            decoded_res = decode_ner_result(viterbi_seq, self.data_dic["id_tag_map"], self.data_dic["raw_dev_data_list"])
            tags_list = [item["tags"].strip().split(" ") for item in self.data_dic["raw_dev_data_list"]]
            f1, precision, recall = computeF1Score(tags_list, decoded_res) 
            print("precision: %0.3f, recall: %0.3f, f1: %0.3f" % (precision, recall, f1))
            save_model_ckpt(sess, config.model_parameters.ckpt_file_path)

    """
    def test(self, sess):
        cnt = 0.0
        test_acc = 0.0
        for x_batch, y_batch in fill_feed_dict(self.data_dic["test_data_x"], 
                                              self.data_dic["test_data_y"], 
                                              config.model_parameters.batch_size):
            acc, prediction = eval_step(self.classifier, sess, (x_batch, y_batch))
            test_acc += acc 
            cnt += 1
        print("Test accuracy : %f %%" % (test_acc / cnt * 100))
    """

if __name__ == "__main__":
    config = ConfigParser(config_file = "../../config/ticketConfig")
    trainer_obj = Trainer(config)
    trainer_obj.train()
