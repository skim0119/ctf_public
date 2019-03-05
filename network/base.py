import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


""" Basic template for building new network module.

Notes:
    Placeholder is indicated by underscore '_' at the end of the variable name
"""


class Deep_layer:
    @staticmethod
    def conv2d_pool(input_layer, channels, kernels, pools, strides,
                    activation=tf.nn.elu, padding='SAME', flatten=False, reuse=False):
        assert len(channels) == len(kernels)
        net = input_layer
        for idx, (ch, kern, pool, stride) in enumerate(zip(channels, kernels, pools, strides)):
            net = layers.conv2d(net, ch, kern, stride,
                                activation_fn=activation,
                                padding=padding,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                reuse=reuse,
                                scope=f'conv_{idx}')
            if pools[idx] > 1:
                net = layers.max_pool2d(net, pool)
        if flatten:
            net = layers.flatten(net)
        return net

    @staticmethod
    def fc(input_layer, hidden_layers, dropout=1.0,
           activation=tf.nn.elu, reuse=False, scope=""):
        net = input_layer
        init = Custom_initializers.variance_scaling()
        for idx, node in enumerate(hidden_layers):
            net = layers.fully_connected(net, int(node),
                                         activation_fn=activation,
                                         scope=f"dense_{idx}"+scope,
                                         weights_initializer=init,
                                         reuse=reuse)
            if idx < len(hidden_layers)-1:
                net = layers.dropout(net,dropout)
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

    @staticmethod
    def variance_scaling():
        return tf.contrib.layers.variance_scaling_initializer(factor = 1.0, mode = "FAN_AVG", uniform = False)
