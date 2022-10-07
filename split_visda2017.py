"""
dataset statistic for long-tail distribution.
"""
import os
import sys
import random
import numpy as np

sys.path.append('.')
import dalib.vision.datasets as datasets

data = 'VisDA2017'

root = 'data/visda-2017'

image_list = {
    "T": "image_list/train.txt",
    "V": "image_list/validation.txt"
}

# Download data
dataset = datasets.__dict__[data]
srcset = dataset(root, task="T", download=True)
trgset = dataset(root, task="V", download=True)
n_class = srcset.num_classes

random.seed(0)
    
for _, list_path in image_list.items():
    txt_path = os.path.join(root, list_path)

    # fetch all pathes and labels
    with open(txt_path, 'r') as f:
        path_list = []
        label_list = []
        for line in f.readlines():
            path, label = line.split()
            label = int(label)
            path_list.append(path)
            label_list.append(label)
    path_list = np.array(path_list)
    label_list = np.array(label_list)

    train_txt = os.path.splitext(txt_path)[0] + '_train.txt'
    test_txt = os.path.splitext(txt_path)[0] + '_test.txt'
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(test_txt):
        os.remove(test_txt)
    for cls_idx in range(n_class):
        idx = np.where(label_list==cls_idx)[0]
        cls_path = path_list[idx]
        random.shuffle(cls_path)
        cls_len = len(cls_path)
        # train split
        train_path = cls_path[:int(cls_len*0.8)]
        with open(train_txt, 'a+') as f:
            for line in train_path:
                f.write(line + ' ' + str(cls_idx) + '\n')
        # test split
        test_path = cls_path[int(cls_len*0.8):]
        with open(test_txt, 'a+') as f:
            for line in test_path:
                f.write(line + ' ' + str(cls_idx) + '\n')