import tensorflow as tf


class config:
    def __init__(self):
        self.model_path = "/home/ste/diskG/StackGAN/model/"

        # 训练集有11532条数据->1个epoch 约 180 steps
        self.batch_size = 64
        self.predict_batch_size = 36
        self.noise_dim = 100
        self.CAnet_dim = 128
        self.init_scale = 0.05
        self.is_training = True
        self.small_image_size = 64
        self.gf_dim = 128
        self.df_dim = 64
        self.batch_norm_decay = 0.9997
        self.epsilon = 0.001

        self.gen_lr = 8e-4
        self.dis_lr = 8e-4
        self.epoch = 600
        self.training_steps = 600 * 180
        self.decay_steps = 5000

        self.data_path = '/home/ste/diskG/CUB_200_2011/'
