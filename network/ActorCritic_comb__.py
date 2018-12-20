import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

import numpy as np
import random

from network.base import base

import utility

class ActorCritic:
    """ Build the graph for A3C model 

    """
    
    def __init__(self, params):
        raise NotImplementedError
    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        raise NotImplementedError
    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care
        about """
        raise NotImplementedError
    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        raise NotImplementedError
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        raise NotImplementedError


class ActorCritic:
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module contains building network and pipelines to sync with global network.
    Global network is expected to have same network structure.
    Actor Critic is implemented with convolution network and fully connected network.
        - LSTM will be added depending on the settings

    Attributes:
        @ Private
        _build_network : 

        @ Public
        run_network :

        update_global :

        pull_global :


    Todo:
        pass

    """
    def __init__(self,
                 in_size,
                 action_size,
                 num_agent,
                 scope,
                 decay_lr=False,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm=0,
                 global_step=None,
                 initial_step=0,
                 lr_a_gamma=1,
                 lr_c_gamma=1,
                 lr_a_step=0,
                 lr_c_step=0,
                 entropy_beta = 0.001,
                 critic_beta = 0.5,
                 sess=None,
                 global_network=None,
                 asynch_training=True,
                 centralize_critic=False):
        """ Initialize AC network and required parameters

        Keyword arguments:
            pass

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:
            * Separate the building trainsequence to separete method.
            * Organize the code with pep8 formating
            
        """

        # Class Environment
        self.sess=sess

        # Parameters & Configs
        self.in_size = in_size
        self.action_size = action_size
        self.grad_clip_norm = grad_clip_norm
        self.num_agent = num_agent
        self.scope = scope
        self.global_step = global_step
        self.asynch_training=asynch_training
        self.centralize_critic=centralize_critic
        
        
        with tf.variable_scope(scope):
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            # Learning Rate Variables
            self.lr_actor = tf.train.exponential_decay(lr_actor,
                                                       self.local_step,
                                                       lr_a_step,
                                                       lr_a_gamma,
                                                       staircase=True,
                                                       name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic,
                                                        self.local_step,
                                                       lr_c_step,
                                                       lr_c_gamma,
                                                       staircase=True,
                                                       name='lr_critic')


            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.a_vars, self.c_vars = self._build_AC_network()

            if scope != 'global':
                self._build_backpropagation()
               
            # Local Network
            # Optimizer
            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')
            if scope != 'global':
                self.action_ = tf.placeholder(shape=[None],dtype=tf.int32, name='action_holder')
                self.action_OH = tf.one_hot(self.action_, action_size)
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')

                with tf.device('/gpu:0'):
                    with tf.name_scope('train'):
                        # Critic (value) Loss
                        td_error = self.td_target_ - self.critic 
                        self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
                        self.critic_loss = tf.reduce_mean(tf.square(td_error),
                                                          name='critic_loss')

                        # Actor Loss
                        obj_func = tf.log(tf.reduce_sum(self.actor * self.action_OH, 1))
                        exp_v = obj_func * self.advantage_ + entropy_beta * self.entropy
                        self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')
                        
                        self.total_loss = critic_beta * self.critic_loss + self.actor_loss

                    with tf.name_scope('local_grad'):
                        a_grads = tf.gradients(self.actor_loss, self.a_vars)
                        c_grads = tf.gradients(self.critic_loss, self.c_vars)
                        if self.grad_clip_norm:
                            a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in a_grads if not grad is None]
                            c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in c_grads if not grad is None]

                    # Sync with Global Network
                    with tf.name_scope('sync'):
                        # Pull global weights to local weights
                        with tf.name_scope('pull'):
                            pull_a_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars, global_network.a_vars)]
                            pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, global_network.c_vars)]
                            self.pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

                        # Push local weights to global weights
                        with tf.name_scope('push'):
                            update_a_op = self.actor_optimizer.apply_gradients(zip(a_grads, global_network.a_vars))
                            update_c_op = self.critic_optimizer.apply_gradients(zip(c_grads, global_network.c_vars))
                            self.update_ops = tf.group(update_a_op, update_c_op)

    def _build_AC_network(self):
        input_shape = [None] + self.in_size
        with tf.variable_scope('actor'):
            self.state_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name='state')
            entry = tf.shape(self.state_input)[0]
            flat_shape = [entry*self.num_agent]+self.in_size[1:]
            bulk_shape = [entry]+self.in_size
            
            net = self.state_input
            net = layers.conv3d(inputs=net,
                                filters=32,
                                kernel_size=[1,5,5]
                                weights_initializer=layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.conv3d(inputs=net,
                                filters=64,
                                kernel_size=[1,3,3]
                                weights_initializer=layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.conv3d(inputs=net,
                                filters=64,
                                kernel_size=[1,2,2]
                                weights_initializer=layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            serial = tf.reshape(net, flat_shape)

            net_actor = layers.fully_connected(serial, 128)
            net_actor = layers.fully_connected(net_actor,
                                               self.action_size,
                                               weights_initializer=layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer(),
                                               activation_fn=tf.nn.softmax)
            self.actor = tf.reshape(net_actor, [-1, self.num_agent, self.action_size])

        with tf.variable_scope('critic'):
            if self.centralize_critic:
                self.mask_ = tf.placeholder(shape=[None, self.num_agent], dtype=tf.float32, name='mask')
                net_critic = tf.stop_gradient(serial)
                net_critic = layers.fully_connected(net_critic, 1,
                                                    weights_initializer=layers.xavier_initializer(),
                                                    biases_initializer=tf.zeros_initializer(),
                                                    activation_fn=None)
                net_critic = tf.reshape(net_critic, [-1, self.num_agent])

                with tf.name_scope('critic_weight'):
                    reweight = tf.get_variable(name='critic_reweight',
                                               shape=[self.num_agent],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(value=1.0/self.num_agent)
                                               )
                    #shift = tf.get_variable(name='critic_shift',
                    #                           shape=[1],
                    #                           dtype=tf.float32,
                    #                           initializer=tf.zeros_initializer
                    #                           )
                    net = tf.multiply(net, reweight) # + shift
                self.critic = tf.reduce_sum(tf.multiply(net, self.mask_), axis=1) # [?, 1]
            else:
                net_critic = tf.stop_gradient(serial)
                net_critic = layers.fully_connected(net_critic, 1,
                                                    weights_initializer=layers.xavier_initializer(),
                                                    biases_initializer=tf.zeros_initializer(),
                                                    activation_fn=None)
                self.critic = tf.reshape(net_critic, [-1, self.num_agent])
    
        # get the parameters of actor and critic networks
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')

        return a_vars, c_vars

    def _build_backpropagation(self):
        with tf.device('/gpu:0'):
            with tf.name_scope('antropy'):
                entropy_frame = -tf.reduce_mean(self.actor * tf.log(self.actor), axis=-1) name='entropy')
                self.entropy = tf.reduce_mean(entropy_frame, name='entropy')

            with tf.name_scope('critic_train'):
                # Critic (value) Loss
                if self.centralize_critic:
                    self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                    td_error = self.td_target_ - self.critic 
                    self.critic_loss = tf.reduce_mean(tf.square(td_error),
                                                      name='critic_loss')
                else:
                    self.td_target_ = tf.placeholder(shape=[None, self.num_agent], dtype=tf.float32, name='td_target_holder')
                    td_error = self.td_target_ - self.critic 
                    self.critic_loss = tf.reduce_mean(tf.square(td_error),
                                                      name='critic_loss')
            with tf.name_scope('actor_train'):
                self.action_ = tf.placeholder(shape=[None, self.num_agent],dtype=tf.int32, name='action_holder')
                self.action_OH = tf.one_hot(self.action_, action_size) # [?, num_agent, action_size]
                self.advantage_ = tf.placeholder(shape=[None, self.num_agent], dtype=tf.float32, name='adv_holder')
                # Actor Loss
                obj_func = tf.log(tf.reduce_sum(self.actor * self.action_OH * self.mask_, 1))
                exp_v = obj_func * self.advantage_ + entropy_beta * self.entropy
                self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')
                
                self.total_loss = critic_beta * self.critic_loss + self.actor_loss

            with tf.name_scope('local_grad'):
                a_grads = tf.gradients(self.actor_loss, self.a_vars)
                c_grads = tf.gradients(self.critic_loss, self.c_vars)
                if self.grad_clip_norm:
                    a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in a_grads if not grad is None]
                    c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in c_grads if not grad is None]

            # Sync with Global Network
            with tf.name_scope('sync'):
                # Pull global weights to local weights
                with tf.name_scope('pull'):
                    pull_a_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars, global_network.a_vars)]
                    pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, global_network.c_vars)]
                    self.pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

                # Push local weights to global weights
                with tf.name_scope('push'):
                    actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor') 
                    critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                    update_a_op = actor_optimizer.apply_gradients(zip(a_grads, global_network.a_vars))
                    update_c_op = critic_optimizer.apply_gradients(zip(c_grads, global_network.c_vars))
                    self.update_ops = tf.group(update_a_op, update_c_op)


    # Choose Action
    def run_network(self, feed_dict):
        a_probs, critics = self.sess.run([self.actor, self.critic], feed_dict)
        actions = [[np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_prob] for a_prob in a_probs]
        return actions, a_probs, critics


    # Pipeline
    def update_global(self, feed_dict):
        self.sess.run(self.update_ops, feed_dict)
        al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)
        
        return al, cl, etrpy

    def pull_global(self):
        self.sess.run(self.pull_op)

