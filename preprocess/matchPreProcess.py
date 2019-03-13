import sys
sys.path.append("../")
import logging
from preprocess.ioUtils import *
from preprocess.tfUtils import *
from setting import *
from config.configParser import ConfigParser


class MatchPreProcess(object):
    def __init__(self):
        pass

    def __generateVocab(self, data_list, use_char=False):
        """
            @brief:构建词表
            @param[in]: data_list 训练集数据，每一项为一个dict
            @param[in]: use_char 是否使用字符表，为True将query按字切分，否则取分词的结果
            @return: vocab dict   
        """
        res = {}
        res[PAD] = len(res)
        res[UNK] = len(res)
        for item in data_list:
            self.__updateVocab(item["query_1"], res, use_char)
            self.__updateVocab(item["query_2"], res, use_char)
        return res

    def __updateVocab(self, query, res, use_char=False):
        word_list = query.replace(" ", "").strip() if use_char else item["query"].strip().split(" ")
        for word in word_list:
            if word in res: continue
            res[word] = len(res)

    def __generateLabelMap(self, data_list, key="intent"):
        """
            @brief:构建label与ID的map
            @param[in]: data_list  训练数据集
            @param[in]: key label对应的key
            @return: label_id_dic, id_label_dic
        """
        label_id_dic = {}
        id_label_dic = {}
        for item in data_list:
            if key not in item:
                logging.warn("corpus does not contains the key: %s" % (key))
                return {}, {}
            if item[key] not in label_id_dic:
                label_id_dic[item[key]] = len(label_id_dic)
                id_label_dic[len(id_label_dic)] = item[key]
        return label_id_dic, id_label_dic
                    
    def __call__(self, config):
        """
            @brief: 预处理类唯一外部接口
            @return: 返回预处理结果的dict 
        """
        #load corpus
        train_data_list = loadJsonData(config.corpus_info.data_path, config.corpus_info.train_data_file)
        dev_data_list = loadJsonData(config.corpus_info.data_path, config.corpus_info.dev_data_file)
        test_data_list = loadJsonData(config.corpus_info.data_path, config.corpus_info.test_data_file)
        print("train_data: %d\t dev_data: %d\t test_data: %d" % \
                 (len(train_data_list), len(dev_data_list), len(test_data_list)))

        #generate vocab 
        vocab = self.__generateVocab(train_data_list, use_char=True)
        saveKeyValueData(vocab, config.corpus_info.data_path + config.corpus_info.vocab_file)    
        
        #generate intent_id_map
        intent_id_map, id_intent_map = self.__generateLabelMap(train_data_list, key="label")
        saveKeyValueData(intent_id_map, config.corpus_info.data_path + config.corpus_info.intent_id_file)
        saveKeyValueData(id_intent_map, config.corpus_info.data_path + config.corpus_info.id_intent_file)

        #encode id matrix
        train_data_x_1 = encodeQuery(train_data_list, vocab, length=25, key="query_1")
        train_data_x_2 = encodeQuery(train_data_list, vocab, length=25, key="query_2")
        train_data_y = encodeLabel(train_data_list, intent_id_map, key="label")

        dev_data_x_1 = encodeQuery(dev_data_list, vocab, length=25, key="query_1")
        dev_data_x_2 = encodeQuery(dev_data_list, vocab, length=25, key="query_2")
        dev_data_y = encodeLabel(dev_data_list, intent_id_map, key="label") 
     
        test_data_x_1 = encodeQuery(test_data_list, vocab, length=25, key="query_1")
        test_data_x_2 = encodeQuery(test_data_list, vocab, length=25, key="query_2")
        test_data_y = encodeLabel(test_data_list, intent_id_map, key="label")

        #package data 
        res_dic = {"raw_train_data_list":train_data_list,
                "raw_dev_data_list":dev_data_list,
                "raw_test_data_list":test_data_list,
                "vocab":vocab,
                "intent_id_map":intent_id_map,
                "id_intent_map":id_intent_map,
                "train_data_x_1":train_data_x_1,
                "train_data_x_2":train_data_x_2,
                "train_data_y":train_data_y,
                "dev_data_x_1":dev_data_x_1,
                "dev_data_x_2":dev_data_x_2,
                "dev_data_y":dev_data_y,
                "test_data_x_1":test_data_x_1,
                "test_data_x_2":test_data_x_2,
                "test_data_y":test_data_y
            }
        return res_dic

if __name__ == "__main__":
    config = ConfigParser(config_file = "../config/atecMatchConfig")
    res_dic = MatchPreProcess()(config)
