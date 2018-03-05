import numpy as np
import scipy.misc
import tensorflow  as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.contrib.gan.python import namedtuples

import StageI
import configuration
import data_provider

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
weights_initializer = tf.initializers.random_normal(mean=0, stddev=0.02)

gen_lr = tf.train.exponential_decay(conf.gen_lr, global_step, conf.decay_steps, 0.5, "generator_learning_rate")
tf.summary.scalar("gen_learning_rate", gen_lr)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5)

dis_lr = tf.train.exponential_decay(conf.dis_lr, global_step, conf.decay_steps, 0.5, "discriminator_learning_rate")
tf.summary.scalar("dis_learning_rate", dis_lr)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr)


# 产生分布的均值与方差
def CAnet(embedding):
    x = slim.fully_connected(embedding, conf.CAnet_dim * 2, activation_fn=tf.nn.leaky_relu)
    mean = x[:, :conf.CAnet_dim]
    log_sigma = x[:, conf.CAnet_dim:]
    return mean, log_sigma


# KL散度的正则化项
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss, name="KL_loss")
        return loss


def residual_blocks(x):
    node0_0 = x
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=weights_initializer):
        node0_1 = slim.conv2d(x, conf.gf_dim * 4, 3)
        node0_1 = slim.conv2d(node0_1, conf.gf_dim * 4, 3, activation_fn=None)

    return tf.nn.relu(tf.add(node0_0, node0_1))


def generator_fn(inputs):
    # inputs is a 2-tuple (noise,embedding)
    gen_img = inputs["gen_img"]
    embedding = inputs["caption"]

    mean, log_sigma = CAnet(embedding)
    c = mean + tf.exp(log_sigma) * tf.truncated_normal(shape=tf.shape(mean))

    s16 = int(conf.small_image_size / 16)
    s8 = int(conf.small_image_size / 8)
    s4 = int(conf.small_image_size / 4)
    s2 = int(conf.small_image_size / 2)

    # encoding gen_image
    x = slim.conv2d(gen_img, conf.gf_dim, 3, weights_initializer=weights_initializer)
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=weights_initializer):
        # 64*64*3
        x = slim.conv2d(x, conf.gf_dim * 2, 4, 2)
        image_encode = slim.conv2d(x, conf.gf_dim * 4, 4, 2)
        # 16*16*512

        # spatial replication
        text_encode = tf.expand_dims(tf.expand_dims(c, 1), 1)
        text_encode = tf.tile(text_encode, [1, s4, s4, 1])

        merge = tf.concat([text_encode, image_encode], 3)
        x = slim.conv2d(merge, conf.gf_dim * 4, 3)

    # 16*16*512
    x = residual_blocks(x)
    x = residual_blocks(x)
    x = residual_blocks(x)
    x = residual_blocks(x)

    # upsampling
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=weights_initializer):
        x = tf.image.resize_nearest_neighbor(x, [s2, s2])
        x = slim.conv2d(x, conf.gf_dim * 2, 3)
        x = tf.image.resize_nearest_neighbor(x, [conf.small_image_size, conf.small_image_size])
        x = slim.conv2d(x, conf.gf_dim, 3)
        x = tf.image.resize_nearest_neighbor(x, [2 * conf.small_image_size, 2 * conf.small_image_size])
        x = slim.conv2d(x, conf.gf_dim // 2, 3)
        x = tf.image.resize_nearest_neighbor(x, [conf.large_image_size, conf.large_image_size])
        x = slim.conv2d(x, conf.gf_dim // 4, 3)

    # 256*256*32
    x = slim.conv2d(x, 3, 3, activation_fn=tf.nn.tanh)
    # 256*256*3

    return x


def discriminator_fn(img, conditioning, weight_decay=2.5e-5):
    embedding = conditioning["caption"]
    # encoding image
    # (256,256,3)
    node1_0 = slim.conv2d(img, conf.df_dim, 4, 2, activation_fn=tf.nn.leaky_relu,
                          weights_initializer=weights_initializer)
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        weights_initializer=weights_initializer):
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 2, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 4, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 8, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 16, 4, 2, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 32, 4, 2, activation_fn=tf.nn.leaky_relu)
        # 4*4*2048
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 16, 1, activation_fn=tf.nn.leaky_relu)
        node1_0 = slim.conv2d(node1_0, conf.df_dim * 8, 1, activation_fn=None)
        # 4*4*512

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
                       normalizer_params=batch_norm_params, activation_fn=tf.nn.leaky_relu,
                       weights_initializer=weights_initializer)
    output = slim.conv2d(feat, 1, s16, s16, activation_fn=None, weights_initializer=weights_initializer)
    return output


def start_train():
    stageI_train_input = data_provider.get_stage_I_train_input_fn()
    condition, real_image = stageI_train_input()
    stageI_gan_model, _ = StageI.get_model_and_loss(condition, real_image)
    conf.is_training = True
    need_to_init = False

    condition, real_image = data_provider.get_stage_II_train_input_fn()()

    with tf.Session() as sess:
        # get_saver
        saver = tf.train.Saver()

        if tf.train.get_checkpoint_state(conf.stageII_model_path):
            saver.restore(sess, tf.train.latest_checkpoint(conf.stageII_model_path))
        else:
            if not tf.train.get_checkpoint_state(conf.stageI_model_path):
                raise FileNotFoundError("StageI model not found!")
            else:
                saver.restore(sess, tf.train.latest_checkpoint(conf.stageI_model_path))
                sI_var = tf.global_variables()
                need_to_init = True
                tf.assign(global_step, 0)

        with tf.variable_scope('Generator', reuse=True):
            gen_img = stageI_gan_model.generator_fn(condition)

        # StageI不参与训练

        param = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
        del param[:]

        gen_input = {"gen_img": gen_img, "caption": condition["caption"]}

        stageII_gan_model, gan_loss = get_model_and_loss(gen_input, real_image)

        gan_train_ops = tfgan.gan_train_ops(
            model=stageII_gan_model,
            loss=gan_loss,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )

        if need_to_init:
            var_to_init = [x for x in tf.global_variables() if x not in sI_var]
            sess.run(tf.initialize_variables(var_to_init))

        train_setp_fn = tfgan.get_sequential_train_steps(namedtuples.GANTrainSteps(1, 10))

        train_writer = tf.summary.FileWriter(conf.stageII_model_path, sess.graph)
        merged = tf.summary.merge_all()
        step = sess.run(global_step)

        with slim.queues.QueueRunners(sess):
            for _ in range(conf.training_steps):
                # test data
                data = sess.run(real_image)
                data = visualize_data(data)
                img = Image.fromarray(data, 'RGB')
                img.show()
                data = sess.run(stageII_gan_model.generator_inputs)
                print(data)
                #
                step = step + 1

                cur_loss, _ = train_setp_fn(sess, gan_train_ops, global_step, {})
                tf.summary.scalar("loss", cur_loss)
                if step % 50 == 0:
                    sumary = sess.run(merged)
                    train_writer.add_summary(sumary, step)

                # save var
                if step % 200 == 0:
                    saver.save(sess, conf.stageI_model_path, global_step)

                # visualize data
                if step % 1000 == 0:
                    gen_data = sess.run(gan_model.generated_data)
                    datas = visualize_data(gen_data)
                    scipy.misc.toimage(datas).save('image/{}.jpg'.format(step))


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
        generator_scope="stageII_generator",
        discriminator_scope="stageII_discriminatior"
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
    condition, real_image = predict_input()
    gan_model, _ = get_model_and_loss(condition, real_image)

    with tf.Session() as sess:
        # get_saver
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(conf.stageI_model_path))

        with tf.variable_scope('Generator', reuse=True):
            pre_img = gan_model.generator_fn(condition)
            imgs = sess.run(pre_img)
            datas = visualize_data(imgs)
            scipy.misc.toimage(datas).save('output/output_img.jpg')


if __name__ == '__main__':
    start_train()
