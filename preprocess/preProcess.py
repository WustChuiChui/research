import sys
sys.path.append("../")
import logging
from preprocess.ioUtils import *
from preprocess.tfUtils import *
from setting import *
from config.configParser import ConfigParser


class PreProcess(object):
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
            word_list = item["query"].replace(" ", "").strip() if use_char else item["query"].strip().split(" ")
            for word in word_list:
                if word not in res:
                    res[word] = len(res)
        return res

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

    def __generateTagsMap(self, data_list, key="tags"):
        """@brief: 生成序列标注TAG与ID的Map
           @param[in]: data_list  训练数据集
           @param[in]: key tag字段在字典中的key
           @return: tag_id_map, id_tag_map
        """
        tag_id_dic = {}
        id_tag_dic = {}
        for item in data_list:
            if key not in item:
                logging.warn("corpus does not contains the key: %s" % (key))
                return {}, {}
            tag_list = item[key].strip().split(" ")
            for tag in tag_list:
                if tag in tag_id_dic:   continue
                tag_id_dic[tag] = len(tag_id_dic)
                id_tag_dic[len(id_tag_dic)] = tag
        return tag_id_dic, id_tag_dic
                    
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
        intent_id_map, id_intent_map = self.__generateLabelMap(train_data_list)
        saveKeyValueData(intent_id_map, config.corpus_info.data_path + config.corpus_info.intent_id_file)
        saveKeyValueData(id_intent_map, config.corpus_info.data_path + config.corpus_info.id_intent_file)
    
        #generate tags_id_map
        tag_id_map, id_tag_map = self.__generateTagsMap(train_data_list)
        saveKeyValueData(tag_id_map, config.corpus_info.data_path + config.corpus_info.tag_id_file)
        saveKeyValueData(id_tag_map, config.corpus_info.data_path + config.corpus_info.id_tag_file)

        #encode id matrix
        train_data_x = encodeQuery(train_data_list, vocab, length=config.model_parameters.max_len) 
        train_data_y = encodeLabel(train_data_list, intent_id_map)
        train_data_tags = encodeTags(train_data_list, tag_id_map, length=config.model_parameters.max_len)    

        dev_data_x = encodeQuery(dev_data_list, vocab, length=config.model_parameters.max_len)
        dev_data_y = encodeLabel(dev_data_list, intent_id_map) 
        dev_data_tags = encodeTags(dev_data_list, tag_id_map, length=config.model_parameters.max_len)
     
        test_data_x = encodeQuery(test_data_list, vocab, length=config.model_parameters.max_len) 
        test_data_y = encodeLabel(test_data_list, intent_id_map)
        test_data_tags = encodeTags(test_data_list, tag_id_map, length=config.model_parameters.max_len)
   
        #package data 
        res_dic = {"raw_train_data_list":train_data_list,
                "raw_dev_data_list":dev_data_list,
                "raw_test_data_list":test_data_list,
                "vocab":vocab,
                "intent_id_map":intent_id_map,
                "id_intent_map":id_intent_map,
                "tag_id_map":tag_id_map,
                "id_tag_map":id_tag_map,
                "train_data_x":train_data_x,
                "train_data_y":train_data_y,
                "train_data_tags":train_data_tags,
                "dev_data_x":dev_data_x,
                "dev_data_y":dev_data_y,
                "dev_data_tags":dev_data_tags,
                "test_data_x":test_data_x,
                "test_data_y":test_data_y,
                "test_data_tags":test_data_tags
            }
        return res_dic

if __name__ == "__main__":
    config = ConfigParser(config_file = "../config/ticketConfig")
    res_dic = PreProcess()(config)
