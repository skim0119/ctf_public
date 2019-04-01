""" Policy Gradient Module

This module contains classes and definition to assist building policy graient model.

Fuctions:
    build_loss (list:Tensor, list:Tensor, list:Tensor): 
        Returns the actor loss, critic loss, and entropy.

Todo:
    * Try to create general policy gradient module
    * Autopep8
    * Docstrings

"""

import tensorflow as tf

import numpy as np

from utility.utils import store_args

class Loss:
    """Loss

    Build function for commonly used loss functions for Policy gradient

    The default is the 'softmax cross-entropy selection' for actor loss and 'TD-difference' for critic error

    """

    @staticmethod
    def softmax_cross_entropy_selection(softmax_logit, action, reward,
                                        td_target, critic,
                                        entropy_beta=0, critic_beta=0,
                                        actor_weight=None, critic_weight=None,
                                        name_scope='loss'):
        with tf.name_scope(name_scope):
            entropy = -tf.reduce_mean(softmax_logit * tf.log(softmax_logit), name='entropy')
            critic_loss = Loss._td_difference(td_target, critic, critic_weight)
            actor_loss = Loss._softmax_cross_entropy(softmax_logit, action, reward, actor_weight)

            if entropy_beta != 0:
                actor_loss += tf.stop_gradient(entropy_beta * entropy)
            if critic_beta != 0:
                actor_loss += tf.stop_gradient(critic_beta * critic_loss)

        return actor_loss, critic_loss, entropy

    @staticmethod
    def _td_difference(target, critic, critic_weight=None):
        if critic_weight is None:
            td_error = target - critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
        else:
            raise NotImplementedError

        return critic_loss

    @staticmethod
    def _softmax_cross_entropy(softmax, action, reward, actor_weight=None):
        if actor_weight is None:
            action_size = tf.shape(softmax)[1]
            action_OH = tf.one_hot(action, action_size)
            obj_func = tf.log(tf.reduce_sum(softmax * action_OH, 1))
            exp_v = obj_func * reward 
            actor_loss = tf.reduce_mean(-exp_v, name='actor_loss') 
        else:
            raise NotImplementedError

        return actor_loss

class Backpropagation:
    """Asynchronous training pipelines"""
    @staticmethod
    def asynch_pipeline(actor_loss, critic_loss,
                        a_vars, c_vars,
                        a_targ_vars, c_targ_vars,
                        lr_actor, lr_critic,
                        name_scope='sync'):
        # Sync with Global Network
        with tf.name_scope(name_scope):
            critic_optimizer = tf.train.AdamOptimizer(lr_critic)
            actor_optimizer = tf.train.AdamOptimizer(lr_actor)

            with tf.name_scope('local_grad'):
                a_grads = tf.gradients(actor_loss, a_vars)
                c_grads = tf.gradients(critic_loss, c_vars)

            with tf.name_scope('pull'):
                pull_a_vars_op = Backpropagation._build_pull(a_vars, a_targ_vars)
                pull_c_vars_op = Backpropagation._build_pull(c_vars, c_targ_vars)
                pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

            with tf.name_scope('push'):
                update_a_op = actor_optimizer.apply_gradients(zip(a_grads, a_targ_vars))
                update_c_op = critic_optimizer.apply_gradients(zip(c_grads, c_targ_vars))
                update_ops = tf.group(update_a_op, update_c_op)

        return pull_op, update_ops

    @staticmethod
    def _build_pull(to_vars, from_vars):
        pull_op = [var.assign(value) for var, value in zip(to_vars, from_vars)]
        return pull_op
