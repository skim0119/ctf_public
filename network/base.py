import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


""" Basic template for building new network module.

Notes:
    Placeholder is indicated by underscore '_' at the end of the variable name
"""


class Deep_layer:
    @staticmethod
    def conv2d_pool(input_layer, channels, kernels, pools,
                    activation=tf.nn.relu, padding='SAME', flatten=False, reuse=False):
        assert len(channels) == len(kernels)
        net = input_layer
        for idx, (ch, kern, pool) in enumerate(zip(channels, kernels, pools)):
            net = layers.conv2d(net, ch, kern,
                                activate_fn=activation,
                                padding=padding,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                reuse=reuse)
            if pools[idx] > 1:
                net = layers.max_pool2d(net, pool)
        if flatten:
            net = layers.flatten(net)
        return net

    @staticmethod
    def fc(input_layer, hidden_layers,
           activation=tf.nn.relu, reuse=False):
        net = input_layer
        for idx, node in enumerate(hidden_layers):
            net = layers.fully_connected(net, node,
                                         activate_fn=activation,
                                         reuse=reuse)
        return net


class Custom_initializers:
    @staticmethod
    def normalized_columns_initializer(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    @staticmethod
    def log_uniform_initializer(mu, std):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.lognormal(mean=mu, sigma=std, size=shape).astype(np.float32)
            return tf.constant(out)
        return _initializer
