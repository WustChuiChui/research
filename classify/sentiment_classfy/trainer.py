import sys, time, os
sys.path.append("../../")
from classify.sentiment_classfy.model import Model
from config.configParser import ConfigParser
from preprocess.preProcess import PreProcess
from preprocess.ioUtils import *
from preprocess.tfUtils import *

class Trainer(object):
    def __init__(self, config):
        print(config)
        self.config = config
        self.classifier = Model(config)
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
                return_dict = self.train_step(self.classifier, sess, (x_batch, y_batch))
            t1 = time.time()
            dev_acc, prediction = self.eval_step(self.classifier, sess, (self.data_dic["dev_data_x"], self.data_dic["dev_data_y"]))
            flag = save_eval_result(self.data_dic["raw_dev_data_list"], prediction, self.data_dic["id_intent_map"], config.corpus_info.dev_res)
            print("Dev accuracy: %.3f  Cost time: %.3f s" % (dev_acc, t1 - t0))
            save_model_ckpt(sess, config.model_parameters.ckpt_file_path)
            ppr_report(self.data_dic["dev_data_y"], prediction, self.data_dic["id_intent_map"])

            test_acc, prediction = self.eval_step(self.classifier, sess, (self.data_dic["test_data_x"], self.data_dic["test_data_y"]))
            print("Test accuracy: %.f " % (test_acc))

    def test(self, sess):
        cnt = 0.0
        test_acc = 0.0
        for x_batch, y_batch in fill_feed_dict(self.data_dic["test_data_x"], 
                                              self.data_dic["test_data_y"], 
                                              config.model_parameters.batch_size):
            acc, prediction = self.eval_step(self.classifier, sess, (x_batch, y_batch))
            test_acc += acc 
            cnt += 1
        print("Test accuracy : %f %%" % (test_acc / cnt * 100))

    def train_step(self, model, sess, batch):
        feed_dict = self.make_train_feed_dict(model, batch)
        to_return = {
                    'train_op': model.train_op,
                    'loss': model.loss,
                    'global_step': model.global_step,
        }
        return sess.run(to_return, feed_dict)

    def make_train_feed_dict(self, model, batch):
        feed_dict = {model.x: batch[0], 
                    model.label: batch[1],
                    model.keep_prob: .5}
        return feed_dict

    def eval_step(self, model, sess, batch): 
        feed_dict = self.make_test_feed_dict(model, batch)
        prediction = sess.run(model.prediction, feed_dict)
        labels = [item.index(max(item)) for item in batch[1].tolist()]
        acc = np.sum(np.equal(prediction, labels)) / len(prediction)
        return acc, prediction

    def make_test_feed_dict(self, model, batch):
        feed_dict = {model.x: batch[0],
                    model.label: batch[1],
                    model.keep_prob: 1.0}
        return feed_dict

if __name__ == "__main__":
    config = ConfigParser(config_file = "../../config/hotelConfig")
    print(type(config))
    sentiment_trainer_obj = Trainer(config)
    sentiment_trainer_obj.train()
