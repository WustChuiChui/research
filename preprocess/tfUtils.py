import sys, os
import numpy as np
from sklearn.utils import shuffle
sys.path.append("../")
from setting import *
import tensorflow as tf
from sklearn.metrics import classification_report

def encodeQuery(data_list, vocab, length=20, padding=PAD):
    res_list = []
    for item in data_list:
        word_id_list = []
        for word in item["query"]:
            if len(word_id_list) >= length: break
            if word in vocab:
                word_id_list.append(vocab[word])
            else:
                word_id_list.append(vocab[UNK])
        for idx in range(len(item["query"]), length):
            word_id_list.append(vocab[PAD])
        res_list.append(word_id_list)
    return np.array(res_list)

def encodeLabel(data_list, label_id_dic, key="intent"):
    res_list = []
    for item in data_list:
        label_id_list = [0] * len(label_id_dic)
        label_id_list[int(label_id_dic[item[key]])] = 1
        res_list.append(label_id_list)
    return np.array(res_list)

#def encodeTags(data_list, tag_id_map, length=20, padding=PAD):

def save_model_ckpt(sess, ckpt_file_path):
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    path += "/" + ckpt_file_path
    if os.path.isdir(path) is False:
        os.makedirs(path)
    print("save ckpt to path: %s" % (path))
    tf.train.Saver().save(sess, path)


def fill_feed_dict(data_X, data_Y, batch_size, need_shuffle=False):
    if need_shuffle:
        data_X, data_Y = shuffle(data_X, data_Y)
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        yield np.array(x_batch), np.array(y_batch)

def make_train_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                model.label: batch[1],
                model.keep_prob: .5}
    return feed_dict                    
                     
def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                model.label: batch[1],
                model.keep_prob: 1.0}
    return feed_dict
                                   
def train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
            'train_op': model.train_op,
            'loss': model.loss,
            'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict) 
    
def eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)
    labels = [item.index(max(item)) for item in batch[1].tolist()]
    acc = np.sum(np.equal(prediction, labels)) / len(prediction)
    return acc, prediction

def ppr_report(encoded_labels, prediction, id_intent_dic):
    max_idxs = list(np.argmax(encoded_labels, 1))
    id_list = sorted(id_intent_dic.items(), key = lambda x:x[1])
    target_names = [item[1] for item in id_list]
    print(classification_report(max_idxs, prediction, target_names=target_names))
