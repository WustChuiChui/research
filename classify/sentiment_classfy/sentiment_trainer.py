import sys, time, os
sys.path.append("../../")
from classify.sentiment_classfy.sentiment_classify import SentimentModel
from config.configParser import ConfigParser
from preprocess.preProcess import PreProcess
from preprocess.ioUtils import *
from preprocess.tfUtils import *

class SentimentTrainer(object):
    def __init__(self, config):
        print(config)
        self.config = config
        self.classifier = SentimentModel(config)
        print("buildGraph")
        self.classifier.buildGraph(config)
        print("buildGraph suceccessful")
        self.classifier.initOptimazer()
        self.data_dic = PreProcess()(config)

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for e in range(config.model_parameters.train_epoch):
            t0 = time.time()
            print("Epoch %d start !" % (e + 1)) 
            for x_batch, y_batch in fill_feed_dict(self.data_dic["train_data_x"], 
                                                  self.data_dic["train_data_y"], 
                                                  config.model_parameters.batch_size,
                                                  need_shuffle=True):
                return_dict = train_step(self.classifier, sess, (x_batch, y_batch))
            t1 = time.time()
            dev_acc, prediction = eval_step(self.classifier, sess, (self.data_dic["dev_data_x"], self.data_dic["dev_data_y"]))
            flag = save_eval_result(self.data_dic["raw_dev_data_list"], prediction, self.data_dic["id_intent_map"], config.corpus_info.dev_res)
            print("Dev accuracy: %.3f  Cost time: %.3f s" % (dev_acc, t1 - t0))
            save_model_ckpt(sess, config.model_parameters.ckpt_file_path)
            ppr_report(self.data_dic["dev_data_y"], prediction, self.data_dic["id_intent_map"])

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

if __name__ == "__main__":
    config = ConfigParser(config_file = "../../config/sentimentConfig")
    print(type(config))
    sentiment_trainer_obj = SentimentTrainer(config)
    sentiment_trainer_obj.train()
