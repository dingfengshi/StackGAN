import tensorflow as tf


class config:
    def __init__(self):
        self.model_path = ""
        self.batch_size = 64
        self.noise_dim = 100
        self.CAnet_dim = 128
        self.init_scale = 0.05
        self.is_training = True
        self.small_image_size = 64
        self.gf_dim = 128
        self.df_dim = 64
        self.batch_norm_decay = 0.9997
        self.epsilon = 0.001

        self.gen_lr = 2e-4
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.gen_lr)
        self.dis_lr = 2e-4
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.dis_lr)

        self.data_path = '/home/ste/diskG/CUB_200_2011/'
