import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

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
                 separate_train=True,
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
        self.in_size = [None]+in_size
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

        # Input/Output
        self.input_tag = self.scope + '/Forward_input/state'
        self.output_tag = self.scope + '/actor/action'

        # RNN Configurations
        self.serial_size = 128  # length of the serial layer (between conv and rnn)
        self.rnn_unit_size = 128  # RNN number of hidden nodes
        self.rnn_num_layers = 3  # RNN number of layers

        with tf.variable_scope(self.scope):
            self._build_placeholders()
            self._build_network()
            if scope == 'global':
                self._build_optimizer()
            else:
                self._build_losses()
                self._build_pipeline()

    def _build_placeholders(self):
        """ Define the placeholders for forward and back propagation """
        with tf.name_scope('Forward_input'):
            self.state_input_ = tf.placeholder(shape=self.in_size, dtype=tf.float32, name='state')
            self.rnn_init_states_ = tf.placeholder(shape=[self.rnn_num_layers, None, self.rnn_unit_size],
                                                   dtype=tf.float32,
                                                   name="rnn_init_states")
            self.seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name="seq_len")

        with tf.name_scope('Backward_input'):
            self.action_ = tf.placeholder(shape=[None, None], dtype=tf.int32, name='action')
            self.reward_ = tf.placeholder(shape=[None, None], dtype=tf.float32, name='reward')
            self.actions_flatten = tf.reshape(self.action_, (-1,))
            self.actions_flat_OH = tf.one_hot(self.actions_flatten, self.action_size)
            self.rewards_flatten = tf.reshape(self.reward_, (-1,))

            self.td_target_ = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name='td_target_holder')
            self.advantage_ = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name='adv_holder')
            self.td_target_flat = tf.reshape(self.td_target_, (-1,))
            self.advantage_flat = tf.reshape(self.advantage_, (-1,))
            # self.likelihood_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_holder')
            # self.likelihood_cumprod_ = tf.placeholder(shape[None], dtype=tf.float32, name='likelihood_cumprod_holder')

    def _build_network(self):
        """ Define network

        It includes convolution network, lstm network, and fully-connected networks
        Separate branch for actor and value

        """
        with tf.variable_scope('actor'):
            # Convolution
            net = tf.reshape(self.state_input_, [-1]+self.in_size[2:])  # Flattening
            net = layers.conv2d(net, 32, [5, 5],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 64, [3, 3],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 64, [2, 2],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            serial_net = layers.flatten(net)
            bulk_shape = tf.stack([tf.shape(self.state_input_)[0],
                                   tf.shape(self.state_input_)[1],
                                   self.serial_size])
            serial_net = layers.fully_connected(serial_net, self.serial_size)
            serial_net = tf.reshape(serial_net, bulk_shape)

            # Recursive Network
            rnn_cell = rnn.GRUCell(self.rnn_unit_size)
            rnn_cells = rnn.MultiRNNCell([rnn_cell] * self.rnn_num_layers)
            rnn_tuple_state = tuple(tf.unstack(self.rnn_init_states_, axis=0))
            rnn_net, self.final_state = tf.nn.dynamic_rnn(rnn_cells,
                                                          serial_net,
                                                          initial_state=rnn_tuple_state,
                                                          sequence_length=self.seq_len_)
            net = tf.reshape(rnn_net, [-1, self.rnn_unit_size])

            # ------------------------------------------------------------------------------------
            # self.rnn_state_in = tf.placeholder(
            #     tf.float32, [self.lstm_layers, 1, self.rnn_steps])
            # self.rnn_init_state = np.zeros((self.lstm_layers, 1, self.rnn_steps))
            # state_per_layer_list = tf.unstack(self.rnn_state_, axis=0)
            # rnn_tuple_state = tuple([holder_ for holder_ in state_per_layer_list])
            # cell = tf.nn.rnn_cell.GRUCell(self.rnn_steps, name='gru_cell')
            # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.lstm_layers)
            # states_series, self.current_state = tf.nn.dynamic_rnn(cell,
            #                                                       tf.expand_dims(net, [0]),
            #                                                       initial_state=rnn_tuple_state,
            #                                                       sequence_length=tf.shape(
            #                                                           self.state_input)[:1]
            #                                                       )
            # net = tf.reshape(states_series[-1], [-1, self.rnn_steps])
            # ------------------------------------------------------------------------------------
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
            # c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            # h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            # self.state_init = [c_init, h_init]
            # self.state_in = (c_in, h_in)
            # rnn_in = tf.expand_dims(hidden, [0])
            # step_size = tf.shape(self.imageIn)[:1]
            # state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            #                                        lstm_cell,
            #                                        rnn_in,
            #                                        initial_state=state_in,
            #                                        sequence_length=step_size,
            #                                        time_major=False
            #                                        )
            # lstm_c, lstm_h = lstm_state
            # self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            # ------------------------------------------------------------------------------------
            # self.trainLength = tf.placeholder(dtype=tf.int32)
            # self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
            # self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
            # self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
            # self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat,
            #                                             cell=rnn_cell,
            #                                             dtype=tf.float32,
            #                                             initial_state=self.state_in,
            #                                             scope=myScope+'_rnn')
            # self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
            # ------------------------------------------------------------------------------------
            # lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size1)
            # self.rnn_serial_length = tf.placeholder(tf.int32)
            # self.rnn_batch_size = tf.placeholder(tf.int32, shape=[])
            # self.rnn_state_in = lstm_cell.zero_state(
            #     self.rnn_batch_size, tf.float32)  # Placeholder
            # # rnn_state_input = tf.contrib.rnn.LSTMStateTuple(*tf.unstack(self.rnn_state_in,axis=0)) # Multicell feature
            # # lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * rnn_multi_layer)
            # rnn_net = tf.reshape(
            #     net, [self.rnn_batch_size, self.rnn_serial_length, hidden_size1], name='serialize')
            # rnn_net, self.rnn_state = tf.nn.dynamic_rnn(cell=lstm_cell,
            #                                             inputs=rnn_net,
            #                                             sequence_length=self.rnn_serial_length,
            #                                             initial_state=self.rnn_state_in,
            #                                             time_major=False
            #                                             )
            # net = tf.reshape(rnn_net, [-1, rnn_hidden_size1])

            self.logit = layers.fully_connected(net,
                                                self.action_size,
                                                weights_initializer=layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                activation_fn=None,
                                                scope='logit')
            self.action = tf.nn.softmax(self.logit, name='action')

        with tf.variable_scope('critic'):
            self.critic = layers.fully_connected(tf.stop_gradient(net),
                                                 1,
                                                 weights_initializer=layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 activation_fn=None)

    def _build_losses(self):
        """ Loss function """
        with tf.name_scope('train'), tf.device('/gpu:0'):
            with tf.name_scope('masker'):
                num_step = tf.shape(self.state_input_)[1]
                self.mask = tf.sequence_mask(self.seq_len, num_step)
                self.mask = tf.reshape(tf.cast(self.mask, tf.float32), (-1,))

            # Entropy
            self.entropy = -tf.reduce_sum(self.action * tf.log(self.action + 1e-8))
            self.entropy = tf.multiply(self.entropy, self.mask)
            self.entropy = tf.reduce_mean(self.entropy, name='entropy')

            # Critic (value) Loss
            td_error = self.td_target_flat - self.critic
            self.critic_loss = tf.reduce_mean(tf.square(td_error*self.mask),  # * self.likelihood_cumprod_),
                                              name='critic_loss')

            # Actor Loss
            obj_func = tf.log(tf.reduce_sum(self.action * self.actions_flat_OH, 1))
            exp_v = obj_func * self.advantage_flat * self.mask + self.entropy_beta * self.entropy
            self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')

            # Total Loss
            self.total_loss = self.critic_beta * self.critic_loss + self.actor_loss

    def _build_pipeline(self):
        """ Define gradient and pipeline to global network """
        if self.separate_train:
            self.a_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor')
            self.c_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')

            with tf.name_scope('local_grad'):
                a_grads = tf.gradients(self.actor_loss, self.a_vars)
                c_grads = tf.gradients(self.critic_loss, self.c_vars)
                if self.grad_clip_norm:
                    a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                               for grad, var in a_grads if grad is not None]
                    c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                               for grad, var in c_grads if grad is not None]

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
                        zip(a_grads, self.global_network.a_vars))
                    update_c_op = self.global_network.critic_optimizer.apply_gradients(
                        zip(c_grads, self.global_network.c_vars))
                    self.update_ops = tf.group(update_a_op, update_c_op)
        else:
            self.graph_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

            with tf.name_scope('local_grad'):
                grads = tf.gradients(self.total_loss, self.graph_vars)
                if self.grad_clip_norm:
                    grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
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
        assert self.scope == 'global'
        if self.separate_train:
            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')
        else:
            self.optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam')

    def feed_forward(self, state, rnn_init_state, seq_len=[1]):
        feed_dict = {self.state_input_: state,
                     self.rnn_init_states_: rnn_init_state,
                     self.seq_len_: seq_len}
        action_prob, critic, final_state = self.sess.run(
            [self.action, self.critic, self.final_state], feed_dict)
        action = [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in action_prob]
        return action, critic, final_state

    def feed_backward(self, states, actions, td_targets, advantages, rnn_init_states, seq_lens):
        feed_dict = {
            self.state_input: states,
            self.action_: actions,
            self.td_target_: td_targets,
            self.advantage_: advantages,
            self.rnn_init_states_: rnn_init_states,
            self.seq_len_: seq_lens
        }
        self.sess.run(self.update_ops, feed_dict)
        al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)

        return al, cl, etrpy

    def pull_global(self):
        self.sess.run(self.pull_ops)

    def get_lstm_initial(self):
        init_state = np.zeros((self.rnn_num_layers, 1, self.rnn_unit_size)
                              )  # 1 for gru state number
        return init_state
