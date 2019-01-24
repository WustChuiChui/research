import sys,logging
import json
from pathlib import Path


"""
@brief: 文件读取预处理脚本文件
@Author: wangjia8@xiaomi.com
"""

def loadKeyValueCorpus(file_name, key="query", value="label"):
    res_list = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            line_data = line.strip().split("\t") 
            if len(line_data) != 2: continue
            if value == "id":
                line_data[1] = int(line_data[1])
            res_list.append({key:line_data[0], value:line_data[1]})
    return res_list


def loadUnlabeledCorpus(file_name):
    res_list = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            line_data = line.strip()
            if line_data == "" or line_data == None:    continue
            res_list.append(line_data)
    return res_list

def saveKeyValueData(data_dic, file_name):
    with open(file_name, "w") as file:
        for key, value in data_dic.items():
            file.writelines(str(key) + "\t" + str(value) + "\n")

def loadJsonData(file_path, file_name):
    file_name = file_path + file_name
    if not Path(file_name).exists():
        logging.warn("%s is not exists." % (file_name))
        return []
    res_list = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            res_list.append(json.loads(line.strip()))
    return res_list

def save_eval_result(data_list, prediction, id_intent_dic, res_file="./dev_result"):
    if len(data_list) != len(prediction):
        logging.warn("dev data size is not equals prediction size")
        return False
    fw = open(res_file, "w")
    fw.write("Flag\tquery\tintent\tpred\n")
    for idx in range(len(data_list)):
        pred = id_intent_dic[prediction[idx]]
        if data_list[idx]["intent"] == pred:
            fw.write("True\t" + data_list[idx]["query"] + "\t" + data_list[idx]["intent"] + "\t" + pred + "\n")
        else:
            fw.write("False\t" + data_list[idx]["query"] + "\t" + data_list[idx]["intent"] + "\t" + pred + "\n")
    return True
