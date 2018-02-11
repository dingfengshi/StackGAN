import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.gan as tfgan
import configuration

tf.reset_default_graph()
conf = configuration.config()
initializer = tf.initializers.random_uniform(-conf.init_scale, conf.init_scale)
batch_norm_params = {
    'decay': conf.batch_norm_decay,
    'epsilon': conf.epsilon,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'is_training': conf.is_training,
    'zero_debias_moving_mean': True
}

global_step = tf.Variable(0, trainable=False, name="global_step")
generator_loss_fn = tfgan.losses.wasserstein_generator_loss
discriminator_loss_fn = tfgan.losses.wasserstein_discriminator_loss


def CAnet(embedding):
    x = slim.fully_connected(embedding, conf.CAnet_dim * 2, activation_fn=tf.nn.leaky_relu,
                             weights_initializer=initializer)
    mean = x[:, :conf.CAnet_dim]
    log_sigma = x[:, conf.CAnet_dim:]
    return mean, log_sigma

def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


def generator_fn(inputs, weight_decay=2.5e-5):
    # inputs is a 2-tuple (noise,embedding)
    noise, embedding = inputs
    mean, log_sigma = CAnet(embedding)
    c = mean + tf.exp(log_sigma) * tf.truncated_normal(shape=tf.shape(mean))

    tf.variable_scope("KL_loss")
    if conf.is_training:
        loss = KL_loss(mean, log_sigma)
    else:
        loss = 0

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
    return (output, embedding)


def discriminator_fn(img, conditioning, weight_decay=2.5e-5):
    _, embedding = conditioning

    s16 = int(conf.small_image_size / 16)
    s8 = int(conf.small_image_size / 8)
    s4 = int(conf.small_image_size / 4)
    s2 = int(conf.small_image_size / 2)

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
    text_encode = tf.tile(text_encode, [1, s16, s16, 1])
    # (:,4,4,128)

    merge = tf.concat([text_encode, image_encode], 3)

    # [1,1]卷积核学习特征

    feat = slim.conv2d(merge, conf.df_dim * 8 + conf.CAnet_dim, 1, normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params, activation_fn=tf.nn.leaky_relu)
    output = slim.conv2d(feat, 1, s16, s16, activation_fn=None)
    return output
