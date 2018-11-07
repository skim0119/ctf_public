import tensorflow as tf
import numpy as np

class base:
    def __init__(self, in_size, action_size, grad_clip_norm, trainable, global_step = None, initial_step = 0, scope='main'):
        self.in_size = in_size
        self.action_size = action_size
        self.grad_clip_norm = grad_clip_norm
        self.trainable = trainable
        self.scope = scope
        
        self.global_step = global_step
        with tf.name_scope(scope):
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')

    def build_train(self):
        pass

    def build_network(self):
        pass

    def build_summarizer(self):
        pass

    #Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer
    
    def log_uniform_initializer(self, mu, std):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.lognormal(mean=mu, sigma=std, size=shape).astype(np.float32)
            return tf.constant(out)
        return _initializer
        