import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from network.base import Deep_layer 


class ActorCritic:
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module contains building network and pipelines to sync with global network.
    Global network is expected to have same network structure.
    Actor Critic is implemented with convolution network and fully connected network.
        - LSTM will be added depending on the settings

    Attributes:
        @ Private
        _build_policy_network :

        @ Public
        run_network :

        update_global :

        pull_global :


    Todo:
        pass

    """
    @store_args
    def __init__(self,
                 local_state_shape,
                 shared_state_shape,
                 action_size,
                 scope,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm=0,
                 entropy_beta=0.001,
                 critic_beta=1.0,
                 explicit_policy=True,
                 sess=None,
                 global_network=None):
        """ Initialize AC network and required parameters

        Keyword arguments:
            explicit_policy: If false, use single critic network and Q value for each action.

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:
            * Separate the building trainsequence to separete method.
            * Organize the code with pep8 formating

        """

        with tf.variable_scope(scope):
            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input = tf.placeholder(shape=local_state_shape, dtype=tf.float32, name='state')
            self.gps_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='gps_state')
            self.goal_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='goal_state')

            # get the parameters of actor and critic networks
            self._build_policy_network(self.state_input)

            self.a_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor')
            self.c_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')

            # Local Network
            if scope == 'global':
                # Optimizer
                self.critic_optimizer = tf.train.AdamOptimizer(
                    self.lr_critic, name='Adam_critic')
                self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')
            else:
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                self.action_OH = tf.one_hot(self.action_, action_size)
                self.td_target_ = tf.placeholder(
                    shape=[None], dtype=tf.float32, name='td_target_holder')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')
#                 self.likelihood_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_holder')
#                 self.likelihood_cumprod_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_cumprod_holder')

                with tf.name_scope('train'), tf.device('/gpu:0'):
                    # Critic (value) Loss
                    td_error = self.td_target_ - self.critic
                    self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
                    self.critic_loss = tf.reduce_mean(tf.square(td_error),  # * self.likelihood_cumprod_),
                                                      name='critic_loss')

                    # Actor Loss
                    obj_func = tf.log(tf.reduce_sum(self.actor * self.action_OH, 1))
                    exp_v = obj_func * self.advantage_
                    self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss') - entropy_beta * self.entropy

                    self.total_loss = critic_beta * self.critic_loss + self.actor_loss - entropy_beta * self.entropy

                with tf.name_scope('local_grad'):
                    a_grads = tf.gradients(self.actor_loss, self.a_vars)
                    c_grads = tf.gradients(self.critic_loss, self.c_vars)
                    if self.grad_clip_norm:
                        a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                   for grad, var in a_grads if grad is not None]
                        c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                   for grad, var in a_grads if grad is not None]

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

    def _build_policy_network(self, input_net):
        with tf.variable_scope('actor'):
            net = Deep_layer.conv2d_pool(input_net, [32,64,64], [5,3,2], [2,2,2],
                                         padding='SAME', flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64, 64],
                                      dropout=1.0)
            goal_array = Deep_layer.fc(input_layer=goal_,
                                       hidden_layers=[64, 64],
                                       dropout=1.0,
                                       reuse=True)

            net = layers.fully_connected(net, 128)

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

    # Choose Action
    def run_network(self, feed_dict):
        a_probs, critic = self.sess.run([self.actor, self.critic], feed_dict)
        return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic
    
    def run_sample(self, feed_dict):
        a_probs = self.sess.run(self.actor, feed_dict)
        return a_probs

    def update_global(self, feed_dict):
        self.sess.run(self.update_ops, feed_dict)
        al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)

        return al, cl, etrpy

    def pull_global(self):
        self.sess.run(self.pull_op)
