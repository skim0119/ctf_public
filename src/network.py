""" Actor-Critic Policy """
import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import random

from utility.utils import store_args

# Network configuration
L2_REGULARIZATION = 0.001


def nn_dense(input, layers_sizes, flatten=False, reuse=False, name=""):
    """Creates a simple layers of fully connected neural network """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


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
                 output_shape,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 entropy_beta=0,  # default: does not include entropy in calculation
                 name=None,
                 sess=None,
                 retrain=True  # if true, reset the graph regardless of saved data
                 ):
        # Class Environment
        self.graph = tf.Graph()

        # Build Graph
        with self.graph.as_default():
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
                var_name = var.name + '_var'
                var_name = var_name.replace(':', '_')
                variable_summary.append(tf.summary.histogram(var_name, var))
            self.var_summary = tf.summary.merge(variable_summary)

            self.loss_summary = self._build_losses()
            self.grad_summary = self._build_pipeline()

            # Save and Restore
            self.saver = tf.train.Saver(self.graph_var, max_to_keep=3)

            # Session
            if sess is None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                            allow_growth=True)
                session_config = tf.ConfigProto(gpu_options=gpu_options)
                # log_device_placement=True)
                self.sess = tf.Session(graph=self.graph, config=session_config)
                for dev in self.sess.list_devices():  # Verbose: list all device
                    print(dev)

            ckpt = tf.train.get_checkpoint_state('./model')
            if retrain:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")
            elif ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

    def save(self, path, global_step=None):
        if global_step is None:
            global_step = self.global_step
        self.saver.save(self.sess, save_path=path, global_step=global_step)

    def _build_placeholders(self):
        """ Define the placeholders """
        self.observations_ = tf.placeholder(tf.float32, self.input_shape, 'observations')
        self.goals_ = tf.placeholder(tf.float32, self.input_shape, 'goals')
        self.actions_ = tf.placeholder(tf.int32, [None], 'actions')
        self.rewards_ = tf.placeholder(tf.float32, [None], 'rewards')
        self.td_targets_ = tf.placeholder(tf.float32, [None], 'td_target')
        self.advantages_ = tf.placeholder(tf.float32, [None], 'baselines')

    def _build_actor_network(self, state_in, goal_in, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            net = tf.concat([state_in, goal_in], 1)
            logits = nn_dense(net, [512, 256, self.output_shape], name)
            dist = tf.nn.softmax(logits, name='action')

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, vars

    def _build_critic_network(self, state_in, goal_in, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            phi = nn_dense(state_in, [512, 256, 1], name)
            psi = nn_dense(goal_in, [512, 256, 1], name, reuse=True)

            critic = np.dot(phi, tf.transpose(psi))
            critic = tf.reshape(critic, [-1])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return critic, vars

    def _build_losses(self):
        with tf.variable_scope('loss'):
            self.global_step = tf.train.get_or_create_global_step()

            with tf.variable_scope('entropy'):
                # self.entropy = -tf.reduce_mean(self.result[0] * tf.log(self.result[0]), name='entropy')
                self.entropy = tf.reduce_mean(self.prob_distribution.entropy())

            with tf.variable_scope('actor'):
                actions_OH = tf.one_hot(self.batch["actions"], self.output_shape)
                obj_func = tf.log(tf.reduce_sum(self.result[0] * actions_OH, 1))
                exp_v = obj_func * self.batch["advantages"]
                self.actor_loss = -tf.reduce_mean(exp_v, name='actor_loss') - self.entropy_beta * self.entropy

            with tf.variable_scope('critic'):
                td_error = self.batch["td_targets"] - self.result[1]
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
                                                               var_list=self.actor_var),
                                      critic_optimizer.minimize(self.critic_loss,
                                                                global_step=self.global_step,
                                                                var_list=self.critic_var)])
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

        action, values = self.sess.run(self.evaluate, feed_dict)
        action, _, values = self.sess.run([self.action_sampling] + self.evaluate, feed_dict)

        return action, values

    def feed_backward(self, experience_buffer, epochs=10, batch_size=256):
        """feed_backward

        :param experience_buffer: (np.array)
            Experience buffer in form of [[state][action][reward][td][advantage]]
        :param epochs: Number of epoch training
        """
        exp_length = len(experience_buffer)

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
                observations, goals, actions, rewards, td_diffs, advantages = experience_buffer[indices]
                for ind in indices:
                    exp_indices.remove(ind)

                feed_dict = {self.observations_: observations,
                             self.goals_: goals,
                             self.actions_: actions,
                             self.rewards_: rewards,
                             self.td_targets_: td_diffs,
                             self.advantages_: advantages}

                summary_ = tf.summary.merge([self.var_summary, self.loss_summary, self.grad_summary])
                train_ops = [summary_, self.global_step, self.train_op]
                summary, step, _ = self.sess.run(train_ops, feed_dict)
        return summary, step
