import numpy as np
import scipy.misc
import tensorflow  as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.estimator import SummaryType
from tensorflow.contrib.learn import RunConfig
import os

import configuration
import data_provider
import losses

tf.reset_default_graph()
conf = configuration.config()
initializer = None
batch_norm_params = {
    'decay': conf.batch_norm_decay,
    'epsilon': conf.epsilon,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'is_training': conf.is_training,
    'zero_debias_moving_mean': True
}

# 训练参数
global_step = tf.train.get_or_create_global_step()
generator_loss_fn = tfgan.losses.modified_generator_loss
discriminator_loss_fn = tfgan.losses.modified_discriminator_loss

gen_lr = tf.train.exponential_decay(conf.gen_lr, global_step, conf.decay_steps, 0.5, "generator_learning_rate")
tf.summary.scalar("gen_learning_rate", gen_lr)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr)

dis_lr = tf.train.exponential_decay(conf.dis_lr, global_step, conf.decay_steps, 0.5, "discriminator_learning_rate")
tf.summary.scalar("dis_learning_rate", dis_lr)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr)


# 产生分布的均值与方差
def CAnet(embedding):
    x = slim.fully_connected(embedding, conf.CAnet_dim * 2, activation_fn=tf.nn.leaky_relu,
                             weights_initializer=initializer)
    mean = x[:, :conf.CAnet_dim]
    log_sigma = x[:, conf.CAnet_dim:]
    return mean, log_sigma


# KL散度的正则化项
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss, name="KL_loss")
        return loss


def generator_fn(inputs, weight_decay=2.5e-5):
    # inputs is a 2-tuple (noise,embedding)
    noise = inputs["noise"]
    embedding = inputs["caption"]
    mean, log_sigma = CAnet(embedding)
    c = mean + tf.exp(log_sigma) * tf.truncated_normal(shape=tf.shape(mean))

    if conf.is_training:
        loss = KL_loss(mean, log_sigma)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    else:
        loss = 0
    tf.summary.scalar("kl_loss", loss)

    s16 = int(conf.small_image_size / 16)
    s8 = int(conf.small_image_size / 8)
    s4 = int(conf.small_image_size / 4)
    s2 = int(conf.small_image_size / 2)

    # (:,228)
    c_n = tf.concat([c, noise], axis=-1, name="generator_input")

    node1_0 = slim.fully_connected(c_n, s16 * s16 * conf.gf_dim * 8, activation_fn=None, normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params)
    node1_0 = tf.reshape(node1_0, [-1, s16, s16, conf.gf_dim * 8])
    # （:,4,4,1024）
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=slim.init_ops.random_uniform_initializer(-conf.init_scale,
                                                                                     conf.init_scale)):
        node1_1 = slim.conv2d(node1_0, conf.gf_dim * 2, 1)
        node1_1 = slim.conv2d(node1_1, conf.gf_dim * 2, 3)
        node1_1 = slim.conv2d(node1_1, conf.gf_dim * 8, 3, activation_fn=None)
        # （:,4,4,1024)
        node1 = tf.add(node1_0, node1_1, "g_node1")
        node1 = tf.nn.relu(node1)
        # （:,4,4,1024)

    node2_0 = tf.image.resize_nearest_neighbor(node1, [s8, s8])
    # (:,8,8,1024)
    node2_0 = slim.conv2d(node2_0, conf.gf_dim * 4, 3, normalizer_fn=slim.batch_norm,
                          normalizer_params=batch_norm_params)
    # (:,8,8,512)

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=slim.init_ops.random_uniform_initializer(-conf.init_scale,
                                                                                     conf.init_scale)):
        node2_1 = slim.conv2d(node2_0, conf.gf_dim, 1)
        node2_1 = slim.conv2d(node2_1, conf.gf_dim, 3)
        node2_1 = slim.conv2d(node2_1, conf.gf_dim * 4, 3, activation_fn=None)
        # (:,8,8,512)
        node2 = tf.add(node2_0, node2_1, "g_node2")
        node2 = tf.nn.relu(node2)
        node2 = tf.image.resize_nearest_neighbor(node2, [s4, s4])
        # (:,16,16,512)
        node2 = slim.conv2d(node2, conf.gf_dim * 2, 3)
        node2 = tf.image.resize_nearest_neighbor(node2, [s2, s2])
        # (:,32,32,256)
        node2 = slim.conv2d(node2, conf.gf_dim, 3)
        node2 = tf.image.resize_nearest_neighbor(node2, [conf.small_image_size, conf.small_image_size])

    output = slim.conv2d(node2, 3, 3, activation_fn=tf.nn.tanh)
    # (:,64,64,3)
    return output


def discriminator_fn(img, conditioning, weight_decay=2.5e-5):
    embedding = conditioning["caption"]
    # encoding image
    # (64,64,3)
    node1_0 = slim.conv2d(img, conf.df_dim, 4, 2, activation_fn=tf.nn.leaky_relu)
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=slim.init_ops.random_uniform_initializer(-conf.init_scale,
                                                                                     conf.init_scale)):
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 2, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 4, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 8, 4, 2, activation_fn=None)
        # (4,4,512)
        node1_1 = slim.conv2d(node1_0, conf.df_dim * 2, 1, activation_fn=tf.nn.leaky_relu)
        node1_1 = slim.conv2d(node1_1, conf.df_dim * 2, 3, activation_fn=tf.nn.leaky_relu)
        node1_1 = slim.conv2d(node1_1, conf.df_dim * 8, 3, activation_fn=None)
        # (4,4,512)
    node1 = tf.add(node1_0, node1_1, "d_node1")
    image_encode = tf.nn.leaky_relu(node1)

    # text embedding compress
    text_encode = slim.fully_connected(embedding, conf.CAnet_dim, activation_fn=tf.nn.leaky_relu)
    # (:,128)
    text_encode = tf.expand_dims(tf.expand_dims(text_encode, 1), 1)

    s16 = int(conf.small_image_size / 16)
    text_encode = tf.tile(text_encode, [1, s16, s16, 1])
    # (:,4,4,128)

    merge = tf.concat([text_encode, image_encode], 3)

    # [1,1]卷积核学习特征

    feat = slim.conv2d(merge, conf.df_dim * 8 + conf.CAnet_dim, 1, normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params, activation_fn=tf.nn.leaky_relu)
    output = slim.conv2d(feat, 1, s16, s16, activation_fn=None)
    return output


def get_estimator():
    # Estimator参数
    config = RunConfig(
        save_summary_steps=100,
        save_checkpoints_steps=300,
        keep_checkpoint_max=5,
        model_dir=conf.model_path
    )

    gan_estimator = tfgan.estimator.GANEstimator(
        model_dir=conf.model_path,
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=losses.get_generator_loss(generator_loss_fn),
        discriminator_loss_fn=discriminator_loss_fn,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        add_summaries=SummaryType.IMAGES,
        config=config
    )

    return gan_estimator


def start_train():
    conf.is_training = True
    train_input = data_provider.get_stage_I_train_input_fn()
    condition, real_image = train_input()

    gan_model, gan_loss = get_model_and_loss(condition, real_image)

    gan_train_ops = tfgan.gan_train_ops(
        model=gan_model,
        loss=gan_loss,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )

    # generator : discrimination = 1:5
    train_setp_fn = tfgan.get_sequential_train_steps(namedtuples.GANTrainSteps(1, 10))

    with tf.Session() as sess:
        # get_saver
        saver = tf.train.Saver()

        if not tf.train.get_checkpoint_state(conf.model_path):
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(conf.model_path))

        train_writer = tf.summary.FileWriter(conf.model_path, sess.graph)
        merged = tf.summary.merge_all()

        with slim.queues.QueueRunners(sess):
            for step in range(conf.training_steps):
                # test data
                # data = sess.run(real_image)
                # data = visualize_data(data)
                # img = Image.fromarray(data, 'RGB')
                # img.show()
                # data = sess.run(gan_model.generator_inputs)
                # print(data)

                cur_loss, _ = train_setp_fn(sess, gan_train_ops, global_step, {})
                tf.summary.scalar("loss", cur_loss)
                if step % 40 == 0:
                    sumary = sess.run(merged)
                    glo_ste = sess.run(global_step)
                    train_writer.add_summary(sumary, glo_ste)

                # save var
                if step % 200 == 0:
                    saver.save(sess, conf.model_path, global_step)

                # visualize data
                if step % 500 == 0:
                    gen_data = sess.run(gan_model.generated_data)
                    glo_ste = sess.run(global_step)
                    datas = visualize_data(gen_data)
                    scipy.misc.toimage(datas).save('image/{}.jpg'.format(glo_ste))


def visualize_data(gen_data):
    batch_size = gen_data.shape[0]
    datas = np.squeeze(np.split(gen_data, batch_size, 0))
    datas = [np.concatenate(datas[i:i + 8]) for i in range(0, 64, 8)]
    datas = np.concatenate(datas, axis=1)
    datas = (datas + 1) / 2 * 255
    datas = np.round(datas)
    datas = datas.astype(np.uint8)
    return datas


def get_model_and_loss(condition, real_image):
    gan_model = tfgan.gan_model(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        real_data=real_image,
        generator_inputs=condition,
    )
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=generator_loss_fn,
        discriminator_loss_fn=discriminator_loss_fn
    )

    return gan_model, gan_loss


def start_predict():
    conf.is_training = False
    predict_input = data_provider.get_stage_I_predict_input_fn()
    gan_estimator = get_estimator()
    prediction_iterable = gan_estimator.predict(predict_input)
    predictions = [prediction_iterable.__next__() for _ in range(conf.predict_batch_size)]
    image_rows = [np.concatenate(predictions[i:i + 6], axis=0) for i in
                  range(0, 36, 6)]
    tiled_images = np.concatenate(image_rows, axis=1)
    tiled_images = (tiled_images + 1) / 2 * 255


if __name__ == '__main__':
    start_train()
