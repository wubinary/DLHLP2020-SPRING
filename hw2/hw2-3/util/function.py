


import torch.nn as nn
import torch


def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    eps = torch.randn(z_mu.shape)
    std = torch.sqrt(torch.exp(z_lv))
    return torch.add(z_mu,torch.multiply(eps,std))
    #with tf.name_scope(name):
    #    eps = tf.random_normal(tf.shape(z_mu))
    #    std = tf.sqrt(tf.exp(z_lv))
    #    return tf.add(z_mu, tf.multiply(eps, std))

