import sys, os, logging
import numpy as np
from sklearn.utils import shuffle
sys.path.append("../")
from setting import *
import tensorflow as tf
from sklearn.metrics import classification_report

def encodeQuery(data_list, vocab, length=20, key="query",padding=PAD):
    res_list = []
    for item in data_list:
        word_id_list = []
        for word in item[key]:
            if len(word_id_list) >= length: break
            if word in vocab:
                word_id_list.append(vocab[word])
            else:
                word_id_list.append(vocab[UNK])
        for idx in range(len(item[key]), length):
            word_id_list.append(vocab[padding])
        res_list.append(word_id_list)
    return np.array(res_list)

def encodeLabel(data_list, label_id_dic, key="intent"):
    res_list = []
    for item in data_list:
        label_id_list = [0] * len(label_id_dic)
        label_id_list[int(label_id_dic[item[key]])] = 1
        res_list.append(label_id_list)
    return np.array(res_list)

def encodeTags(data_list, tag_id_map, length=20, padding="O"):
    res_list = []
    for item in data_list:
        if "tags" not in item:
            logging.warn("corpus does not contains tags for NER task, return []")
            return []
        tag_list = item["tags"].strip().split(" ")
        word_tag_list = []
        for tag in tag_list:
            if len(word_tag_list) >= length:    break
            if tag in tag_id_map:
                word_tag_list.append(tag_id_map[tag])
            else:
                logging.error("Invalid tag in token %s " % (tag))
                exit(-1)
        for idx in range(len(word_tag_list), length):
            word_tag_list.append(tag_id_map[padding])
        res_list.append(word_tag_list)
    return np.array(res_list)

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

def ppr_report(encoded_labels, prediction, id_intent_dic):
    max_idxs = list(np.argmax(encoded_labels, 1))
    id_list = sorted(id_intent_dic.items(), key = lambda x:x[0], reverse=False)
    target_names = [item[1] for item in id_list]
    print(classification_report(max_idxs, prediction, target_names=target_names))

def decode_ner_result(verterbi_seq, id_tag_map, raw_data_list, max_len=20):
    res = []
    if len(verterbi_seq) != len(raw_data_list): 
        logging.warn("verterbi_seq size is not equals to the size of raw_data_list.")    
        return res
    for idx in range(len(verterbi_seq)):
        query_tag_list = []
        query = raw_data_list[idx]["query"]
        for index in range(len(query)):
            if index >= max_len: break
            query_tag_list.append(id_tag_map[int(verterbi_seq[idx][index])])
        res.append(query_tag_list)
    return res

def __splitTagType(tag, split_token="_"):
    s = tag.split(split_token)
    #if len(s) > 2 or len(s) == 0:
    if len(s) > 3 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        #tagType = s[1]
        tagType = " ".join(s[1:])
    return tag, tagType

def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart = False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd = False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd

def computeF1Score(correct_slots, pred_slots, split_token="-"):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        print("correct_slot: %s \t\tpred_slot: %s" % (correct_slot, pred_slot))
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c, split_token=split_token)
            predTag, predType = __splitTagType(p, split_token=split_token)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                     __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                     (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall
