""" Actor-Critic Policy """
import tensorflow as tf
import tensorflow.layers as layers

import numpy as np
import random

from utility.utils import store_args

# Network configuration
L2_REGULARIZATION = 0.001


def nn_dense(input_tensor, layers_sizes, flatten=False, reuse=tf.AUTO_REUSE, name="fc"):
    """Creates a simple layers of fully connected neural network """
    input_tensor = layers.dense(inputs=input_tensor, units=512, reuse=False)
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input_tensor = layers.dense(inputs=input_tensor,
                                    units=size,
                                    activation=activation,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    reuse=reuse,
                                    name=name + '_' + str(i))
    if flatten:
        # assert layers_sizes[-1] == 1
        input_tensor = tf.reshape(input_tensor, [-1,])
    return input_tensor


class Initializer:
    @staticmethod
    def normalized_columns_init(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer


class ActorCritic:
    """ TensorFlow implementation of Actor Critic Algorithm

    Module contains network structure and pipeline for dataset.
    Code is adapted to the environment CtF.

    It takes 2D multi-channel image as an input, and gives 5 probability of action
    """

    @store_args
    def __init__(self,
                 input_shape,
                 goal_shape,
                 output_size,
                 sess,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 entropy_beta=0,  # default: does not include entropy in calculation
                 scope=None,
                 retrain=True  # if true, reset the graph regardless of saved data
                 ):

        # Build Graph
        with tf.variable_scope(scope):
            self._build_placeholders()

            # Build network
            self.policy, self.policy_vars = self._build_actor_network(state_in=self.observations_,
                                                                      goal_in=self.goals_,
                                                                      name='actor')
            self.critic, self.critic_vars = self._build_critic_network(state_in=self.observations_,
                                                                       goal_in=self.goals_,
                                                                       name='critic')
            self.result = [self.policy, self.critic]
            self.graph_var = self.policy_vars + self.critic_vars
            self.prob_distribution = tf.distributions.Categorical(probs=self.policy)
            self.action_sampling = self.prob_distribution.sample()

            # Build Summary and Training Operations
            variable_summary = []
            for var in self.graph_var:
                print(var)
                var_name = var.name + '_var'
                var_name = var_name.replace(':', '_')
                variable_summary.append(tf.summary.histogram(var_name, var))
            self.var_summary = tf.summary.merge(variable_summary)

            self.loss_summary = self._build_losses()
            self.grad_summary = self._build_pipeline()

    def _build_placeholders(self):
        """ Define the placeholders """
        self.observations_ = tf.placeholder(tf.float32, self.input_shape, 'observations')
        self.goals_ = tf.placeholder(tf.float32, self.goal_shape, 'goals')
        self.actions_ = tf.placeholder(tf.int32, [None], 'actions')
        self.rewards_ = tf.placeholder(tf.float32, [None], 'rewards')
        self.td_targets_ = tf.placeholder(tf.float32, [None], 'td_target')
        self.advantages_ = tf.placeholder(tf.float32, [None], 'baselines')

    def _build_actor_network(self, state_in, goal_in, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            net = tf.concat([state_in, goal_in], 1)
            logits = nn_dense(net, [512, 256, self.output_size])
            dist = tf.nn.softmax(logits, name='action')

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+'/'+name)
        return dist, vars

    def _build_critic_network(self, state_in, goal_in, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            phi = nn_dense(state_in, [256, 256, 1], name+'_phi')
            psi = nn_dense(goal_in, [256, 256, 1], name+'_psi')

            critic = np.dot(phi, tf.transpose(psi))
            critic = tf.reshape(critic, [-1])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+'/'+name)
        return critic, vars

    def _build_losses(self):
        with tf.variable_scope('loss'):
            self.global_step = tf.train.get_or_create_global_step()

            with tf.variable_scope('entropy'):
                # self.entropy = -tf.reduce_mean(self.result[0] * tf.log(self.result[0]), name='entropy')
                self.entropy = tf.reduce_mean(self.prob_distribution.entropy())

            with tf.variable_scope('actor'):
                #actions_OH = tf.one_hot(self.actions_, self.output_size)
                #obj_func = tf.log(tf.reduce_sum(self.policy * actions_OH, 1))
                #exp_v = obj_func * self.advantages_
                #self.actor_loss = -tf.reduce_mean(exp_v, name='actor_loss') - self.entropy_beta * self.entropy

                self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy, labels=self.actions_)
                self.actor_loss = tf.reduce_mean(self.cross_entropy_loss, name='actor_loss') - self.entropy_beta * self.entropy

            with tf.variable_scope('critic'):
                td_error = self.td_targets_ - self.critic
                self.critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            summaries = []
            summaries.append(tf.summary.scalar("actor_loss", self.actor_loss))
            summaries.append(tf.summary.scalar("critic_loss", self.critic_loss))
            summaries.append(tf.summary.scalar("entropy", self.entropy))

            return tf.summary.merge(summaries)

    def _build_pipeline(self):
        with tf.variable_scope('train'):
            actor_optimizer = tf.train.AdamOptimizer(self.lr_actor)
            critic_optimizer = tf.train.AdamOptimizer(self.lr_critic)
            self.train_op = tf.group([actor_optimizer.minimize(self.actor_loss,
                                                               var_list=self.policy_vars),
                                      critic_optimizer.minimize(self.critic_loss,
                                                                global_step=self.global_step,
                                                                var_list=self.critic_vars)])
            grads = actor_optimizer.compute_gradients(self.actor_loss)

            summaries = []
            for grad, var in grads:
                if grad is None:
                    continue
                var_name = var.name + '_grad'
                var_name = var_name.replace(':', '_')
                summaries.append(tf.summary.histogram(var_name, grad))
            return tf.summary.merge(summaries)

    def feed_forward(self, states, goals):
        feed_dict = {self.observations_: states,
                     self.goals_: goals}

        action, values = self.sess.run(self.result, feed_dict)
        action, _, values = self.sess.run([self.action_sampling] + self.result, feed_dict)

        return action, values

    def feed_backward(self, experience_buffer, epochs=10, batch_size=256):
        """feed_backward

        :param experience_buffer: (np.array)
            Experience buffer in form of [[state][action][reward][td][advantage]]
        :param epochs: Number of epoch training
        """
        # Draw batch samples
        observations, actions, rewards, td_diffs, advantages, global_goal, goal_id, goal_played = experience_buffer
        exp_length = len(observations)

        for ep in range(epochs):
            exp_indices = list(range(exp_length))
            remaining_length = exp_length
            while remaining_length:
                if remaining_length < batch_size:
                    indices = exp_indices
                    remaining_length = 0
                else:
                    indices = random.sample(exp_indices, batch_size)
                    remaining_length -= batch_size

                for ind in indices:
                    exp_indices.remove(ind)

                feed_dict = {self.observations_: observations[indices],
                             self.goals_: goal_played[indices],
                             self.actions_: actions[indices],
                             self.rewards_: rewards[indices],
                             self.td_targets_: td_diffs[indices],
                             self.advantages_: advantages[indices]}

                summary_ = tf.summary.merge([self.var_summary]) #, self.loss_summary, self.grad_summary])
                train_ops = [summary_, self.train_op]
                summary, _ = self.sess.run(train_ops, feed_dict)
        return summary

    def feed_backward_meta(self, experience_buffer, epochs=10, batch_size=256):
        """feed_backward

        :param experience_buffer: (np.array)
            Experience buffer in form of [[state][action][reward][td][advantage]]
        :param epochs: Number of epoch training
        """
        # Draw batch samples
        observations, actions, rewards, td_diffs, advantages, global_goal, goal_id, goal_played = experience_buffer
        exp_length = len(observations)

        for ep in range(epochs):
            exp_indices = list(range(exp_length))
            remaining_length = exp_length
            while remaining_length:
                if remaining_length < batch_size:
                    indices = exp_indices
                    remaining_length = 0
                else:
                    indices = random.sample(exp_indices, batch_size)
                    remaining_length -= batch_size

                # Draw batch samples
                for ind in indices:
                    exp_indices.remove(ind)

                feed_dict = {self.observations_: observations[indices],
                             self.goals_: global_goal[indices],
                             self.actions_: goal_id[indices],
                             self.rewards_: rewards[indices],
                             self.td_targets_: td_diffs[indices],
                             self.advantages_: advantages[indices]}

                summary_ = tf.summary.merge([self.var_summary]) #, self.loss_summary, self.grad_summary])
                train_ops = [summary_, self.train_op]
                summary, _ = self.sess.run(train_ops, feed_dict)
        return summary
