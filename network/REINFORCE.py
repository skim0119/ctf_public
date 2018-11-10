import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random

from network.base import base

import utility

class REINFORCE(base):
    ###
    #
    ###
    def __init__(self, lr, in_size,action_size, grad_clip_norm, trainable=False, scope='main', board=True, sess = None):
        #super(REINFORCE,self).__init__(in_size, action_size, grad_clip_norm, trainable)
        self.in_size = in_size
        self.action_size = action_size
        self.grad_clip_norm = grad_clip_norm
        self.trainable = trainable
        self.lr = lr
        
        self.sess = sess
           
        # Set Parameters
        with tf.name_scope('Network_Param'):
            self.input_shape=tf.constant(in_size[1:3], name='input_shape')
            self.action_sizes = tf.constant(action_size, name='output_size')

        ## Build tensorflow Graph
        self.build_network()
                
        # Update Operations
        if trainable:
            with tf.name_scope(scope+'/train'):
                self.entropy = -tf.reduce_mean(self.output * tf.log(self.output+1e-8), name='entropy') # measure action diversity
                self.responsible_outputs = tf.reduce_sum(self.output * self.action_OH, 1)
                self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward_holder)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
                self.grads = [tf.clip_by_norm(grad, 50) for grad in self.grads]

                with tf.name_scope('grad_holders'):
                    self.grad_holders = [(tf.Variable(var, trainable=False, dtype=tf.float32, name=var.op.name+'_buffer'), var) for var in tf.trainable_variables()]
                self.accumulate_gradient = tf.group([tf.assign_add(a[0],b[0]) for a,b in zip(self.grad_holders, self.grads)]) # add gradient to buffer
                self.clear_batch = tf.group([tf.assign(a[0],a[0]*0.0) for a in self.grad_holders])
                self.update_batch = self.optimizer.apply_gradients(self.grad_holders, self.local_step) ## update and increment step
        if board:
            self.build_summary()



    def build_network(self):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_input = tf.placeholder(shape=self.in_size,dtype=tf.float32, name='state')
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name='actions')
        self.action_OH = tf.one_hot(self.action_holder, self.action_size)
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name='reward')
        
        ## Convolution Layer
        layer = slim.conv2d(self.state_input, 32, [5,5], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='VALID')
        layer = slim.max_pool2d(layer, [2,2])
        layer = slim.conv2d(layer, 64, [3,3], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='VALID')
        layer = slim.max_pool2d(layer, [2,2])
        layer = slim.conv2d(layer, 64, [2,2], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='VALID')
        layer = slim.flatten(layer)

        ## Fully Connected Layer
        layer = layers.fully_connected(layer, 128, 
                            weights_initializer=self.normalized_columns_initializer(0.001),
                            activation_fn=tf.nn.relu)
        self.dense = layers.fully_connected(layer, self.action_size,
                            weights_initializer=self.normalized_columns_initializer(0.001),
                            activation_fn=None)
        self.output = tf.nn.softmax(self.dense, name='action')

    def build_summary(self):
        # Summary
        # Histogram output
        with tf.variable_scope('debug_parameters'):
            tf.summary.histogram('output', self.output)
            tf.summary.histogram('actor', self.dense)     
            tf.summary.histogram('action', self.action_holder)
        
        # Graph summary Loss
        with tf.variable_scope('summary'):
            tf.summary.scalar(name='total_loss', tensor=self.loss)
            tf.summary.scalar(name='Entropy', tensor=self.entropy)
        
        with tf.variable_scope('weights_bias'):
            # Histogram weights and bias
            for var in slim.get_model_variables():
                tf.summary.histogram(var.op.name, var)
                
        with tf.variable_scope('gradients'):
            # Histogram Gradients
            for var, grad in zip(slim.get_model_variables(), self.gradients):
                tf.summary.histogram(var.op.name+'/grad', grad[0])

