import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import time
import numpy as np
import random

import base

# Training Related
max_ep = 150
update_frequency = 50
batch_size = 2000
experience_size=10000

# Parameters
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
#Create a directory to save episode playback gifs to
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
    
LEARNING_RATE = 1e-3

class REINFORCE(base):
    def __init__(self, lr, in_size,action_size, grad_clip_norm, trainable=False):
        super().__init__()
        
        # Set Parameters
        with tf.name_scope('Network_Param'):
            self.input_shape=tf.constant(np.array([VISION_dX,VISION_dY]), name='input_shape')
            self.action_size = tf.constant(action_size, name='output_size')
        
        # Set Constants
        self.grad_clip_norm=grad_clip_norm
        

    def build_network(self, scope):
        with tf.variable_scope(scope):
            #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
            self.state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')
            self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
            self.action_OH = tf.one_hot(self.action_holder, action_size)
            self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name='reward')
            
            ## Convolution Layer
            layer = slim.conv2d(self.state_input, 32, [5,5], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID',
                                scope='conv1')
            layer = slim.max_pool2d(layer, [2,2])
            layer = slim.conv2d(layer, 64, [3,3], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID',
                                scope='conv2')
            layer = slim.max_pool2d(layer, [2,2])
            layer = slim.conv2d(layer, 64, [2,2], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID',
                                scope='conv3')
            layer = slim.flatten(layer)

            ## Fully Connected Layer
            layer = layers.fully_connected(layer, 128, 
                                weights_initializer=normalized_columns_initializer(0.001),
                                activation_fn=tf.nn.relu)
            self.dense = layers.fully_connected(layer, action_size,
                                weights_initializer=normalized_columns_initializer(0.001),
                                activation_fn=None,
                                scope='output_fc')
            self.output = tf.nn.softmax(self.dense, name='action')

        def build_train(self, scope):
            # Update Operations
            with tf.name_scope('train'):
                self.entropy = -tf.reduce_mean(self.output * tf.log(self.output+1e-8), name='entropy')
                self.responsible_outputs = tf.reduce_sum(self.output * self.action_OH, 1)
                self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward_holder)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                self.gradients = self.optimizer.compute_gradients(self.loss)
                self.grads = [tf.clip_by_norm(grad, 50) for grad in self.gradients]
                
                self.grad_holders = [(tf.Variable(var, trainable=False, dtype=tf.float32, name=var.op.name+'_buffer'), var)
                                     for var in tf.trainable_variables()]
                self.update_batch = self.optimizer.apply_gradients(self.grad_holders)
                self.accumulate_gradient = tf.group([tf.assign_add(a[0],b[0]) for a,b in zip(self.grad_holders, self.grads)]) # add gradient to buffer
                self.clear_batch = tf.group([tf.assign(a[0],a[0]*0.0) for a in self.grad_holders])
                                            
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

global_step = tf.Variable(0, trainable=False, name='global_step') # global step
increment_global_step_op = tf.assign(global_step, global_step+1)
merged = tf.summary.merge_all()

# Setup Save and Restore Network
saver = tf.train.Saver(tf.global_variables())
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Load Model : ", ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    print("Initialized Variables")    
