import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib

import numpy as np


class ActorCritic:
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module contains building network and pipelines to sync with global network.
    Global network is expected to have same network structure.
    Actor Critic is implemented with convolution network and fully connected network.
        - LSTM will be added depending on the settings

    Todo:
        pass
    """

    def __init__(self,
                 in_size,
                 action_size,
                 scope,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm=0,
                 global_step=None,
                 critic_beta=0.25,
                 entropy_beta=0.01,
                 sess=None,
                 global_network=None,
                 separate_train=False,
                 sequence_maxlen=8
                 ):
        """ Initialize AC network

        Keyword arguments:
            pass

        Features:
            separate_train:
                False, use single optimizer to minimize total_loss = critic_loss + actor_loss
                    - Use lr_actor as a total learning rate
                True, use two optimizer and two learning rate for critic and actor losses

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
        self.sess = sess

        # Parameters & Configs
        self.in_size = [None] + in_size
        self.action_size = action_size
        self.scope = scope
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.grad_clip_norm = grad_clip_norm
        self.global_step = global_step
        self.critic_beta = critic_beta
        self.entropy_beta = entropy_beta
        self.global_network = global_network
        self.separate_train = separate_train
        self.sequence_maxlen = sequence_maxlen

        # Input/Output
        self.input_tag = self.scope + '/Forward_input/state'
        self.output_tag = self.scope + '/actor/action'

        # RNN Configurations
        self.rnn_type = 'GRU'
        self.serial_size = 256# length of the serial layer (between conv and rnn)
        self.rnn_unit_size = 256# RNN number of hidden nodes
        self.rnn_num_layers = 1  # RNN number of layers

        with tf.variable_scope(self.scope):
            self._build_placeholders()
            self._build_network()
            self._build_optimizer()
            if scope != 'global':
                self._build_losses()
                self._build_pipeline()

    def _build_placeholders(self):
        """ Define the placeholders for forward and back propagation """
        # Forward
        self.state_input_ = tf.placeholder(shape=self.in_size, dtype=tf.float32, name='state')
        self.seq_len_ = tf.placeholder(shape=(None,), dtype=tf.int32, name="seq_len")
        if self.rnn_type == 'GRU':
            self.rnn_init_states_ = tf.placeholder(shape=[self.rnn_num_layers, None, self.rnn_unit_size],
                                                   dtype=tf.float32,
                                                   name="rnn_init_states")
        else:
            self.rnn_init_states_ = tf.placeholder(shape=[self.rnn_num_layers, None, self.rnn_unit_size],
                                                   dtype=tf.float32,
                                                   name="rnn_init_states")

        # Backward
        self.actions_ = tf.placeholder(shape=[None, None], dtype=tf.int32, name='action_hold')
        self.actions_flat_ = tf.reshape(self.actions_, (-1,))
        self.actions_OH = tf.one_hot(self.actions_flat_, self.action_size)

        self.td_target_ = tf.placeholder(shape=[None, None], dtype=tf.float32, name='td_target_holder')
        self.td_target_flat_ = tf.reshape(self.td_target_, (-1,))
        self.advantage_ = tf.placeholder(shape=[None, None], dtype=tf.float32, name='adv_holder')
        self.advantage_flat_ = tf.reshape(self.advantage_, (-1,))
        self.likelihood_ = tf.placeholder(shape=[None], dtype=tf.float32, name='likelihood_holder')
        self.likelihood_cumprod_ = tf.placeholder(shape=[None], dtype=tf.float32, name='likelihood_cumprod_holder')

    def _build_network(self):
        """ Define network
        It includes convolution network, lstm network, and fully-connected networks
        Separate branch for actor and value
        """
        with tf.variable_scope('actor'):
            # Parameter
            batch_size, seq_length = tf.shape(self.state_input_)[0], tf.shape(self.state_input_)[1]
            bulk_shape = tf.stack([batch_size, seq_length, self.serial_size])

            # Convolution
            net = tf.reshape(self.state_input_, [-1] + self.in_size[-3:])
            # net = layers.conv2d(net, 32, [5, 5],
            #                     weights_initializer=layers.xavier_initializer_conv2d(),
            #                    biases_initializer=tf.zeros_initializer(),
            #                    padding='SAME')
            #net = layers.max_pool2d(net, [2, 2])
            # net = layers.conv2d(net, 64, [3, 3],
            #                    weights_initializer=layers.xavier_initializer_conv2d(),
            #                    biases_initializer=tf.zeros_initializer(),
            #                    padding='SAME')
            #net = layers.max_pool2d(net, [2, 2])
            # net = layers.conv2d(net, 64, [2, 2],
            #                    weights_initializer=layers.xavier_initializer_conv2d(),
            #                    biases_initializer=tf.zeros_initializer(),
            #                    padding='SAME')
            serial_net = layers.flatten(net)
            serial_net = layers.fully_connected(serial_net, self.serial_size, activation_fn=tf.nn.elu)

            # Recursive Network
            rnn_net = tf.reshape(serial_net, bulk_shape)
            if self.rnn_type == 'GRU':
                #rnn_cells = tf.contrib.cudnn_rnn.CudnnGRU(self.rnn_num_layers, self.rnn_unit_size)
                #rnn_net, self.final_state = rnn_cells(rnn_net)
                #rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.8)
                rnn_cells = rnn.MultiRNNCell([rnn.GRUCell(self.rnn_unit_size) for _ in range(self.rnn_num_layers)])
                rnn_tuple_state = tuple(tf.unstack(self.rnn_init_states_, axis=0))  # unstack by rnn layer
                rnn_net, self.final_state = tf.nn.dynamic_rnn(rnn_cells,
                                                              rnn_net,
                                                              initial_state=rnn_tuple_state
                                                              #sequence_length=self.seq_len_
                                                              )
                rnn_net = tf.reshape(rnn_net, (-1, self.rnn_unit_size))
            else:
                # Multi RNN Cell is not yet implemented
                self.lstm_cell = rnn.BasicLSTMCell(self.rnn_unit_size)
                self.lstm_cell = rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=0.8)
                self.lstm_cells = rnn.MultiRNNCell([self.lstm_cell for _ in range(self.rnn_num_layer)])
                self.rnn_init_states_ = self.lstm_cells.zero_state(batch_size, tf.float32)  # placeholder
                rnn_tuple_state = tuple(tf.unstack(self.rnn_init_states_, axis=0))
                rnn_net, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                        rnn_net,
                                                        initial_state=rnn_tuple_state,
                                                        #sequence_length=self.seq_len_,
                                                        )
                self.final_state = lstm_state
                # lstm_c, lstm_h = lstm_state
                # self.final_state = (lstm_c[:1, :], lstm_h[:1, :])
                rnn_net = tf.reshape(rnn_net, (-1, self.rnn_unit_size))

            self.logit = layers.fully_connected(rnn_net,
                                                self.action_size,
                                                weights_initializer=layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                activation_fn=None,
                                                )
            self.action = tf.nn.softmax(self.logit, name='action')

            # ------------------------------------------------------------------------------------
            # lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size1)
            # self.rnn_serial_length = tf.placeholder(tf.int32)
            # self.rnn_batch_size = tf.placeholder(tf.int32, shape=[])
            # self.rnn_state_in = lstm_cell.zero_state(
            #     self.rnn_batch_size, tf.float32)  # Placeholder
            # # rnn_state_input = tf.contrib.rnn.LSTMStateTuple(*tf.unstack(self.rnn_state_in,axis=0)) # Multicell feature
            # # lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * rnn_multi_layer)
            # rnn_net, self.rnn_state = tf.nn.dynamic_rnn(cell=lstm_cell,
            #                                             inputs=rnn_net,
            #                                             sequence_length=self.rnn_serial_length,
            #                                             initial_state=self.rnn_state_in,
            #                                             time_major=False
            #                                             )
            # net = tf.reshape(rnn_net, [-1, rnn_hidden_size1])

        with tf.variable_scope('critic'):
            critic_net = layers.fully_connected(tf.stop_gradient(serial_net),
                                                1,
                                                activation_fn=None)
            self.critic = tf.reshape(critic_net, [-1, ])  # column to row

        if self.separate_train:
            self.a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
            self.c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        else:
            self.graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.ad_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)

    def _build_losses(self):
        """ Loss function """
        ppo_epsilon = 0.2
        with tf.name_scope('train'):
            with tf.name_scope('masker'):
                self.mask = tf.sequence_mask(lengths=self.seq_len_, dtype=tf.float32)
                self.mask_flat = tf.reshape(self.mask, (-1,))

            # Entropy
            self.entropy = -tf.reduce_mean(self.action * tf.log(self.action), name='entropy')

            # Critic (value) Loss
            td_error = self.td_target_flat_ - self.critic
            self.critic_loss = tf.reduce_mean(tf.square(td_error), # * self.mask_flat),  # * self.likelihood_cumprod_,
                                              name='critic_loss')

            # Actor Loss
            # ppo
            # exp_v_a = self.likelihood_ * self.advantage_flat_ * self.mask_flat
            # exp_v_b = tf.clip_by_value(self.likelihood_, 1 - ppo_epsilon, 1 + ppo_epsilon) * self.advantage_flat_ * self.mask_flat
            # self.actor_loss = -tf.reduce_mean(tf.minimum(exp_v_a, exp_v_b), name='actor_loss') - self.entropy_beta * self.entropy
            obj_func = tf.log(tf.reduce_sum(self.action * self.actions_OH, 1))
            exp_v = obj_func * self.advantage_flat_ * self.likelihood_ # * self.mask_flat
            self.actor_loss = -tf.reduce_mean(exp_v, name='actor_loss')# - self.entropy_beta * self.entropy

            # Total Loss
            self.total_loss = self.critic_beta * self.critic_loss + self.actor_loss

    def _build_pipeline(self):
        """ Define gradient and pipeline to global network """
        if self.separate_train:
            with tf.name_scope('local_grad'):
                self.a_grads = tf.gradients(self.actor_loss, self.a_vars)
                self.c_grads = tf.gradients(self.critic_loss, self.c_vars)
                if self.grad_clip_norm:
                    self.a_grads = [tf.clip_by_value(grad, -10, 10)
                                    for grad in self.a_grads if grad is not None]
                    self.c_grads = [tf.clip_by_value(grad, -10, 10)
                                    for grad in self.c_grads if grad is not None]

            # Sync with Global Network
            with tf.name_scope('sync'):
                # Pull global weights to local weights
                with tf.name_scope('pull'):
                    pull_a_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(
                        self.a_vars, self.global_network.a_vars)]
                    pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(
                        self.c_vars, self.global_network.c_vars)]
                    self.pull_ops = tf.group(pull_a_vars_op, pull_c_vars_op)
                # Push local weights to global weights
                with tf.name_scope('push'):
                    update_a_op = self.global_network.actor_optimizer.apply_gradients(
                        zip(self.a_grads, self.global_network.a_vars))
                    update_c_op = self.global_network.critic_optimizer.apply_gradients(
                        zip(self.c_grads, self.global_network.c_vars))
                    self.update_ops = tf.group(update_a_op, update_c_op)
        else:
            with tf.name_scope('local_grad'):
                grads = tf.gradients(self.total_loss, self.graph_vars)
                if self.grad_clip_norm:
                    grads = [(tf.clip_by_value(grad, -10, 10), var)
                             for grad, var in grads if grad is not None]

            # Sync with Global Network
            with tf.name_scope('sync'):
                # Pull global weights to local weights
                with tf.name_scope('pull'):
                    self.pull_ops = [local_var.assign(glob_var)
                                     for local_var, glob_var in zip(self.graph_vars, self.global_network.graph_vars)]
                # Push local weights to global weights
                with tf.name_scope('push'):
                    self.update_ops = self.global_network.optimizer.apply_gradients(
                        zip(grads, self.global_network.graph_vars))

    def _build_optimizer(self):
        """ Optimizer """
        if self.separate_train:
            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic)
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.lr_critic)

    def feed_forward(self, state, rnn_init_state, seq_len=[1]):
        feed_dict = {self.state_input_: state,
                     self.rnn_init_states_: rnn_init_state,
                     self.seq_len_: seq_len}
        action_prob, critic, final_state = self.sess.run(
            [self.action, self.critic, self.final_state], feed_dict)
        action = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in action_prob]
        return action, critic.tolist(), final_state

    def feed_backward(self, states, actions, td_targets, advantages, rnn_init_states, seq_lens, retrace_lambda=None):
        if retrace_lambda is not None:
            # Get likelihood of global with states
            actions_flat = np.reshape(actions, (-1,))
            feed_dict = {self.state_input_: states,
                         self.rnn_init_states_: rnn_init_states,
                         self.seq_len_: seq_lens}
            current_prob = self.sess.run(self.action, feed_dict)
            soft_prob = self.sess.run(self.global_network.action,
                                      feed_dict={self.global_network.state_input_: states,
                                                 self.global_network.rnn_init_states_: rnn_init_states,
                                                 self.global_network.seq_len_: seq_lens})
            target_policy = np.array([p[action] for p, action in zip(soft_prob, actions_flat)])
            behavior_policy = np.array([p[action] for p, action in zip(current_prob, actions_flat)])
            likelihood = []
            likelihood_cumprod = []
            running_prob = 1.0
            for pi, beta in zip(target_policy, behavior_policy):
                ratio = retrace_lambda * min(1.0, pi / (beta + 1e-10))
                running_prob *= ratio
                likelihood.append(ratio)
                likelihood_cumprod.append(running_prob)
            likelihood = np.array(likelihood)
            likelihood_cumprod = np.array(likelihood_cumprod)
        else:
            likelihood = np.ones_like(advantages)
            likelihood_cumprod = np.ones_like(advantages)

        feed_dict = {
            self.state_input_: states,
            self.actions_: actions,
            self.td_target_: td_targets,
            self.advantage_: advantages,
            self.rnn_init_states_: rnn_init_states,
            self.seq_len_: seq_lens,
            self.likelihood_: likelihood,
            self.likelihood_cumprod_: likelihood_cumprod
        }
        self.sess.run(self.update_ops, feed_dict)
        al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)

        return al, cl, etrpy

    def pull_global(self):
        self.sess.run(self.pull_ops)

    def get_lstm_initial(self, batch_size=1):
        if self.rnn_type == 'GRU':
            # 1 for gru state number
            init_state = np.zeros((self.rnn_num_layers, batch_size, self.rnn_unit_size))
        else:
            c_init = np.zeros((1, self.lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, self.lstm_cell.state_size.h), np.float32)
            init_state = [c_init, h_init]
        return init_state
