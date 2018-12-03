import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

import numpy as np
import random

from network.base import base

import utility

class ActorCritic():
    def __init__(self,
                 in_size,
                 action_size,
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
                 lstm_network=False,
                 separate_train=False):
        """ Initialize AC network and required parameters

        Keyword arguments:
            pass

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:
            pass
            
        """

        # Class Environment
        self.sess=sess

        # Parameters & Configs
        self.in_size = in_size
        self.action_size = action_size
        self.grad_clip_norm = grad_clip_norm
        self.scope = scope
        self.global_step = global_step
        self.lstm_network = lstm_network
        self.separate_train = separate_train
        
        # Dimensions
        self.lstm_layers = 1
        
        with tf.variable_scope(scope):
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            # Learning Rate Variables
            self.lr_actor = tf.train.exponential_decay(lr_actor, self.local_step,
                                                       lr_a_step, lr_a_gamma, staircase=True, name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic, self.local_step,
                                                       lr_c_step, lr_c_gamma, staircase=True, name='lr_critic')


            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')

            # get the parameters of actor and critic networks
            self.build_network()
            if self.separate_train:
                self.a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor')
                self.c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')
                
            else:
                self.graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
                
            # Local Network
            if scope == 'global':
                if self.separate_train:
                    # Optimizer
                    self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                    self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')
                else:
                    # Optimizer
                    self.optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam')
            else:
                self.action_ = tf.placeholder(shape=[None],dtype=tf.int32, name='action_holder')
                self.action_OH = tf.one_hot(self.action_, action_size)
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')
#                 self.likelihood_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_holder')
#                 self.likelihood_cumprod_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_cumprod_holder')

                with tf.device('/gpu:0'):
                    with tf.name_scope('train'):
                        # Critic (value) Loss
                        td_error = self.td_target_ - self.critic 
                        self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
                        self.critic_loss = tf.reduce_mean(tf.square(td_error), #* self.likelihood_cumprod_),
                                                          name='critic_loss')

                        # Actor Loss
                        obj_func = tf.log(tf.reduce_sum(self.actor * self.action_OH, 1))
                        exp_v = obj_func * self.advantage_ + entropy_beta * self.entropy
                        self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')
                        
                        self.total_loss = critic_beta * self.critic_loss + self.actor_loss

                    if self.separate_train:
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
                                update_a_op = global_network.actor_optimizer.apply_gradients(zip(a_grads, global_network.a_vars))
                                update_c_op = global_network.critic_optimizer.apply_gradients(zip(c_grads, global_network.c_vars))
                                self.update_ops = tf.group(update_a_op, update_c_op)
                            
                    else:
                        with tf.name_scope('local_grad'):
                            grads = tf.gradients(self.total_loss, self.graph_vars)
                            if self.grad_clip_norm:
                                grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in grads if not grad is None]

                        # Sync with Global Network
                        with tf.name_scope('sync'):
                            # Pull global weights to local weights
                            with tf.name_scope('pull'):
                                self.pull_op = [local_var.assign(glob_var)
                                                for local_var, glob_var in zip(self.graph_vars, global_network.graph_vars)]

                            # Push local weights to global weights
                            with tf.name_scope('push'):
                                self.update_ops = global_network.optimizer.apply_gradients(zip(grads, global_network.graph_vars))
                            
    def build_network(self, init_mean=-1.0, init_std=1.0):
        with tf.variable_scope('actor'):
            net = self.state_input
            net = layers.conv2d(net , 32, [5,5],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2,2])
            net = layers.conv2d(net, 64, [3,3],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2,2])
            net = layers.conv2d(net, 64, [2,2],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.flatten(net)
            net = layers.fully_connected(net, 128)

            if self.lstm_network:
                rnn_steps = 16
                self.rnn_state_ = tf.placeholder(tf.float32, [self.lstm_layers, 1, rnn_steps])
                self.rnn_init_state = np.zeros((self.lstm_layers, 1, rnn_steps))
                #self.rnn_state_ = tf.placeholder(tf.float32, [self.lstm_layers, 2, 1, rnn_steps])
                #self.rnn_init_state = np.zeros((self.lstm_layers, 2, 1, rnn_steps))

                state_per_layer_list = tf.unstack(self.rnn_state_, axis=0)
                rnn_tuple_state = tuple(
                    [holder_ for holder_ in state_per_layer_list]
                )
                #rnn_tuple_state = tuple(
                #    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                #     for idx in range(self.lstm_layers)]
                #)

                cell = tf.nn.rnn_cell.GRUCell(rnn_steps, name='gru_cell')
                #cell = tf.nn.rnn_cell.LSTMCell(rnn_steps, forget_bias=1, name='lstm_cell')
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.lstm_layers)
                states_series, self.current_state = tf.nn.dynamic_rnn(cell,
                                                                      tf.expand_dims(net, [0]),
                                                                      initial_state=rnn_tuple_state,
                                                                      sequence_length=tf.shape(self.state_input)[:1]
                                                                      )
                net = tf.reshape(states_series[-1], [-1, rnn_steps])
                
            self.actor = layers.fully_connected(net,
                                                self.action_size,
                                                weights_initializer=layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                activation_fn=tf.nn.softmax)

        with tf.variable_scope('critic'):
            self.critic = layers.fully_connected(net,
                                                 1,
                                                 weights_initializer=layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 activation_fn=None)
            self.critic = tf.reshape(self.critic, [-1])

    # Update global network with local gradients
    def update_global(self, feed_dict):
        self.sess.run(self.update_ops, feed_dict)
        al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)
        
        return al, cl, etrpy

    def pull_global(self):
        self.sess.run(self.pull_op)

    # Choose Action
    def run_network(self, feed_dict):
        if self.lstm_network:
            a_probs, critic, rnn_state = self.sess.run([self.actor, self.critic, self.current_state], feed_dict)
            return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic, rnn_state
        else:
            a_probs, critic = self.sess.run([self.actor, self.critic], feed_dict)
            return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic, None  
