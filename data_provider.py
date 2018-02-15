import tensorflow as tf
import numpy as np

import configuration

conf = configuration.config()


def parse_data(example):
    features = {
        "image": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "image_shape": tf.FixedLenFeature(shape=[3], dtype=tf.int64),
        "caption": tf.VarLenFeature(tf.float32),
        "caption_shape": tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    }
    par_exm = tf.parse_single_example(example, features=features)
    image = tf.reshape(tf.decode_raw(par_exm["image"], tf.uint8), par_exm["image_shape"])
    caption = tf.reshape(par_exm["caption"].values, par_exm["caption_shape"])
    return image, caption


def map_Stage_I(example):
    image, caption = parse_data(example)
    resized_image = tf.image.resize_images(image, [conf.small_image_size, conf.small_image_size])
    cap_num = tf.shape(caption.shape[0].value)
    choice = np.random.choice(range(cap_num))
    single_caption = caption[choice, :]
    return resized_image, caption


def get_train_input_fn():
    data_path = "/home/ste/diskG/StackGAN/CUB/train.tfrecord"

    def train_input_fn():
        with tf.device("/cpu:0"):
            dataset = tf.data.TFRecordDataset(data_path)
            dataset = dataset.map(map_Stage_I)
            dataset = dataset.shuffle(buffer_size=512)
            dataset = dataset.repeat(conf.epoch)
            dataset = dataset.batch(conf.batch_size)
            iterator = dataset.make_one_shot_iterator()
            image, caption = iterator.get_next()
            noise = tf.random_normal(conf.noise_dim)
            return (noise, caption), image
