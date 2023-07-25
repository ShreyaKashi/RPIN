import random
import json


random.seed(100)

with open("./mcs_preprocess/scene_name_dict.json", "r", encoding="utf-8") as f:
    content = json.load(f)


test=random.sample(list(content.keys()), 35)
train=list(set(list(content.keys()))-set(test))


train_set={}
for i in range(len(train)):
    train_set[train[i]]=content[train[i]]

test_set={}
for i in range(len(test)):
    test_set[test[i]]=content[test[i]]

print(len(train_set))
print(len(test_set))

with open('./mcs_preprocess/train_set_dict.json', 'w') as f:
    json.dump(train_set, f)

with open('./mcs_preprocess/test_set_dict.json', 'w') as f:
    json.dump(test_set, f)