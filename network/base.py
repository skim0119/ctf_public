import tensorflow as tf
import numpy as np


""" Basic template for building new network module.

Notes:
    Placeholder is indicated by underscore '_' at the end of the variable name
"""


class Base:
    def __init__(self):
        pass

    def _build_placeholders(self):
        raise NotImplementedError

    def _build_embedding(self):
        raise NotImplementedError

    def _build_loss(self):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError


class Custom_initializers:
    def normalized_columns_initializer(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    def log_uniform_initializer(mu, std):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.lognormal(mean=mu, sigma=std, size=shape).astype(np.float32)
            return tf.constant(out)
        return _initializer
