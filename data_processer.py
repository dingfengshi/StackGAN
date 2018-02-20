import os
from math import floor

from PIL import Image
import torchfile

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

images_list_path = "/home/ste/diskG/StackGAN/CUB/images.txt"
captions_path = "/home/ste/diskG/StackGAN/CUB/caption/"
images_path = "/home/ste/diskG/StackGAN/CUB/images/"


def preprocess_caption():
    embedding_script_path = "/home/ste/diskG/StackGAN/embedding/get_embedding.lua"

    img_list = get_img_list(images_list_path)

    for each in img_list:
        filenames = captions_path + each + ".t7"
        queries = captions_path + each + ".txt"
        command = "filenames={} queries={} /home/ste/diskE/torch/install/bin/th {}".format(filenames, queries,
                                                                                           embedding_script_path)
        print(os.system(command))


def cut_pic():
    bound_boxes_path = "/home/ste/diskG/StackGAN/CUB/bounding_boxes.txt"
    img_list = get_img_list(images_list_path)
    idx = 0
    with open(bound_boxes_path) as fin:
        for each in fin.readlines():
            elem = each.rstrip().split()
            path = images_path + img_list[idx] + ".jpg"
            save_path = images_path + img_list[idx] + "_crop.jpg"
            idx = idx + 1
            img = Image.open(path)
            x = int(float(elem[1]))
            y = int(float(elem[2]))
            width = int(float(elem[3]))
            height = int(float(elem[4]))
            img = img.crop((x, y, x + width, y + height))
            img.save(save_path)


def get_tfrecord():
    train_test_split_path = "/home/ste/diskG/StackGAN/CUB/train_test_split.txt"
    train_save_path = "/home/ste/diskG/StackGAN/CUB/train.tfrecord"
    test_save_path = "/home/ste/diskG/StackGAN/CUB/test.tfrecord"
    test_list_num = 256

    img_list = get_img_list(images_list_path)
    train_list = []
    test_list = []

    n = 0
    with open(train_test_split_path) as fin:
        for each in fin.readlines():
            line = each.rstrip().split()
            idx = line[0]
            is_train = line[1]
            if is_train == "1" or n >= test_list_num:
                train_list.append(img_list[int(idx) - 1])
            else:
                test_list.append(img_list[int(idx) - 1])
                n = n + 1

    # print(train_list.__len__())
    # print(test_list.__len__())

    # train_dataset
    writer = tf.python_io.TFRecordWriter(train_save_path)
    write_tfrecord(train_list, writer)
    writer.close()

    # test dataset
    writer = tf.python_io.TFRecordWriter(test_save_path)
    write_tfrecord(test_list, writer)
    writer.close()


def write_tfrecord(data_list, writer):
    for each in data_list:
        img_path = images_path + each + "_crop.jpg"
        cap_path = captions_path + each + ".t7"
        img = Image.open(img_path)
        if len(img.getbands()) != 3:
            continue
        _cap = torchfile.load(cap_path)
        # list has caption_num vector
        cap = _cap.fea_txt
        cap = np.array(cap)
        cap_num = cap.shape[0]
        cap = cap.flatten()
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "image_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(img.size) + [3])),
            "caption": tf.train.Feature(float_list=tf.train.FloatList(value=cap)),
            "caption_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=[cap_num, 1024]))
        }))
        writer.write(example.SerializeToString())


def get_img_list(path):
    img_list = []
    with open(path) as fin:
        for eachline in fin.readlines():
            path = eachline.rstrip().split()[1]
            img_list.append(path[:-4])

    return img_list




if __name__ == '__main__':
    get_tfrecord()
