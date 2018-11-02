import os

import signal

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
%matplotlib inline

import time
import gym
import gym_cap
import gym_cap.envs.const as CONST
import numpy as np
import random

# the modules that you can use to generate the policy. 
import policy.random
import policy.roomba
import policy.policy_RL
import policy.zeros

# Data Processing Module
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards

%load_ext autoreload
%autoreload 2

TRAIN_NAME='B4R4_Rzero_AC_MonteCarlo'
LOG_PATH='./logs/'+TRAIN_NAME
MODEL_PATH='./model/' + TRAIN_NAME
GPU_CAPACITY=0.2 # gpu capacity in percentage (adjustable)

# Training Related
max_ep = 150
update_frequency = 50
batch_size = 2000
experience_size=10000

# Saving Related
save_network_frequency = 1000
save_stat_frequency = 100
moving_average_step = 100

# Parameters
LEARNING_RATE = 1e-3
gamma = 0.99
MAP_SIZE = 20
VISION_RANGE = 9
VISION_dX, VISION_dY = 2*VISION_RANGE+1, 2*VISION_RANGE+1

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
#Create a directory to save episode playback gifs to
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
    
class agent():
    def __init__(self, lr, in_size,action_size, grad_clip_norm):
        self.grad_clip_norm=grad_clip_norm
        
        with tf.name_scope('Network_Param'):
            self.input_shape=tf.constant(np.array([VISION_dX,VISION_dY]), name='input_shape')
        
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.action_OH = tf.one_hot(self.action_holder, action_size)
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name='reward')
        
        layer = slim.conv2d(self.state_input, 32, [5,5], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            scope='conv1')
        layer = slim.max_pool2d(layer, [2,2])
        layer = slim.conv2d(layer, 64, [3,3], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            scope='conv2')
        layer = slim.max_pool2d(layer, [2,2])
        layer = slim.conv2d(layer, 64, [2,2], activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            scope='conv3')
        layer = slim.flatten(layer)
        layer = layers.fully_connected(layer, 128, 
                                    activation_fn=tf.nn.relu)
        self.dense = layers.fully_connected(layer, action_size,
                                    activation_fn=None,
                                    scope='output_fc')
        self.output = tf.nn.softmax(self.dense, name='action')
        
        # Update Operations
        with tf.name_scope('train'):
            self.responsible_outputs = tf.reduce_sum(self.output * self.action_OH, 1)
            self.loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.reward_holder)
            '''self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.dense, labels=self.action_holder)*self.reward_holder)'''
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
        
        with tf.variable_scope('weights_bias'):
            # Histogram weights and bias
            for var in slim.get_model_variables():
                tf.summary.histogram(var.op.name, var)
                
        with tf.variable_scope('gradients'):
            # Histogram Gradients
            for var, grad in zip(slim.get_model_variables(), self.gradients):
                tf.summary.histogram(var.op.name+'/grad', grad[0])

tf.reset_default_graph() # Clear the Tensorflow graph.
myAgent = agent(lr=LEARNING_RATE,in_size=[None,VISION_dX,VISION_dY,6],action_size=5,grad_clip_norm=50) #Load the agent.
global_step = tf.Variable(0, trainable=False, name='global_step') # global step
increment_global_step_op = tf.assign(global_step, global_step+1)
merged = tf.summary.merge_all()

# Launch the session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.Session()

ma_reward = MA(moving_average_step)
ma_length = MA(moving_average_step)
ma_captured = MA(moving_average_step)

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
    
    
    
