import json

train_list = list(open("train_data", "r").readlines())
train_list = [item.replace(" ", "").strip().split("\t") for item in train_list]
train_list = [{"query":item[0], "intent":item[1]} for item in train_list]

dev_list = list(open("dev_data", "r").readlines())
dev_list = [item.replace(" ", "").strip().split("\t") for item in dev_list]
dev_list = [{"query":item[0], "intent":item[1]} for item in dev_list]

test_list = list(open("test_data", "r").readlines())
test_list = [item.replace(" ", "").strip().split("\t") for item in test_list]
test_list = [{"query":item[0], "intent":item[1]} for item in test_list]

print(len(train_list))
print(len(dev_list))
print(len(test_list))

fw = open("train_data_2", "w")
for item in train_list:
    fw.write(json.dumps(item, ensure_ascii=False))
    fw.write("\n")

fw = open("dev_data_2", "w")
for item in dev_list:
    fw.write(json.dumps(item, ensure_ascii=False))
    fw.write("\n")

fw = open("test_data_2", "w")
for item in test_list:
    fw.write(json.dumps(item, ensure_ascii=False))
    fw.write("\n")
