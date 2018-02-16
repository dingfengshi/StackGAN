import tensorflow as tf
import tensorflow.contrib.gan as tfgan
import configuration

conf = configuration.config()


def generator_loss_with_kl_KL_divergence(loss_fn):
    def new_loss_fn(gan_model, **kargs):
        kl_loss = tf.get_default_graph().get_tensor_by_name("Generator/KL_divergence/KL_loss:0")
        return kl_loss + loss_fn(gan_model)

    return new_loss_fn


def get_generator_loss(loss_fn):
    if conf.is_training:
        return generator_loss_with_kl_KL_divergence(loss_fn)
    else:
        return loss_fn
