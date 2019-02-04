""" Actor-Critic Policy """
import tensorflow as tf
import tensorflow.layers as layers

import numpy as np

# Network configuration
MINIBATCH_SIZE = 512
L2_REGULARIZATION = 0.001


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

    def __init__(self,
                 input_shape,
                 output_shape,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 entropy_beta=0.01,
                 name=None,
                 sess=None,
                 ):
        # Class Environment
        self.graph = tf.Graph()
        self.sess = sess
        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                        allow_growth=True)
            session_config = tf.ConfigProto(gpu_options=gpu_options,
                                            log_device_placement=True)
            self.sess = tf.Session(graph=self.graph, config=Session_config))
        self.name = name

        # Parameters & Configs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_beta = entropy_beta

        # Build Graph
        with self.graph.as_default(): 
            self._build_placeholders()
            self._build_dataset()

            # Policy Network for Backpropagation
            policy, policy_vars = self._build_actor_network(self.batch["observations"], 'actor')
            critic, critic_vars = self._build_critic_network(self.batch["observations"], 'critic')
            self.result = [policy, critic]
            self.actor_var = policy_vars
            self.critic_var = baseline_vars
            self.graph_var = policy_vars + baseline_vars
            self.prob_distribution = tf.distributions.Categorical(probs=policy)

            # Policy Network for Forward Pass (Same Weight)
            policy_eval, _ = self._build_actor_network(self.observations_,
                                                       'actor',
                                                       reuse=True,
                                                       batch_size=1)
            critic_eval, _ = self._build_critic_network(self.observations_,
                                                        'critic',
                                                        reuse=True,
                                                        batch_size=1)
            self.evaluate = [policy_eval, critic_eval]
            self.eval_distribution = tf.distributions.Categorical(probs=policy_eval)

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


    def save(self, path, global_step=None):
        if global_step is None:
            global_step = self.global_step
        self.saver.save(self.sess, save_path=path, global_step=global_step)

    def restore(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load Model : ", ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("Initialized Variables")

    def _build_placeholders(self):
        """ Define the placeholders """
        self.observations_ = tf.placeholder(tf.float32, self.input_shape, 'observations')
        self.actions_ = tf.placeholder(tf.int32, [None], 'actions')
        self.rewards_ = tf.placeholder(tf.float32, [None], 'rewards')
        self.td_targets_ = tf.placeholder(tf.float32, [None], 'td_target')
        self.advantages_ = tf.placeholder(tf.float32, [None], 'baselines')

    def _build_dataset(self):
        """ Use the TensorFlow Dataset API """
        with tf.device('/cpu:0'):
            self.dataset = tf.data.Dataset.from_tensor_slices({"observations": self.observations_,
                                                               "actions": self.actions_,
                                                               "rewards": self.rewards_,
                                                               "td_targets": self.td_targets_,
                                                               "advantages": self.advantages_})
                                          .shuffle(buffer_size=1000)
                                          .batch(MINIBATCH_SIZE, drop_remainder=True)
            self.iterator = self.dataset.make_initializable_iterator()
            self.batch = self.iterator.get_next()

    def _build_actor_network(self, state_in, name, reuse=False, batch_size=MINIBATCH_SIZE):
        w_reg = None

        with tf.variable_scope(name, reuse=reuse):
            net = state_in
            net = layers.conv2d(net, 32, [5, 5],
                                activation_fn=tf.nn.relu,
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 64, [3, 3],
                                activation_fn=tf.nn.relu,
                                padding='SAME')
            net = layers.flatten(net)
            net = layers.fully_connected(net, 128)

            logits = layers.dense(net, self.output_shape,
                                  kernel_initializer=Initializer.normalized_columns_init(0.01),
                                  kernel_regularizer=w_reg,
                                  name='pi_logits')
            dist = tf.nn.softmax(logits, name='action')
            policy_dist = tf.reshape(dist, [batch_size, self.output_shape])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return policy_dist, vars

    def _build_critic_network(self, state_in, name, reuse=False, batch_size=MINIBATCH_SIZE):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REGULARIZATION)

        with tf.variable_scope(name, reuse=reuse):
            net = state_in
            net = layers.conv2d(net, 32, [5, 5],
                                activation_fn=tf.nn.relu,
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 64, [3, 3],
                                activation_fn=tf.nn.relu,
                                padding='SAME')
            net = layers.flatten(net)

            critic = layers.dense(net, 1,
                                  kernel_initializer=Initializer.normalized_columns_init(1.0),
                                  kernel_regularizer=w_reg, name="critic_out")
            critic = tf.reshape(critic, [batch_size])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return critic, vars

    def _build_losses(self):
        with tf.variable_scope('loss'):
            self.global_step = tf.train.get_or_create_global_step()

            with tf.variable_scope('entropy'):
                #self.entropy = -tf.reduce_mean(self.result[0] * tf.log(self.result[0]), name='entropy')
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
                var_name = var.name + '_grad'
                var_name = var_name.replace(':', '_')
                summaries.append(tf.summary.histogram(var_name, grad))
            return tf.summary.merge(summaries)

    def feed_forward(self, states):
        feed_dict = {self.observations_: states}

        action, _, values = self.sess.run([self.eval_distribution]+self.evaluate, feed_dict)

        return action, values

    def feed_backward(self, experience_buffer, epochs=10):
        """feed_backward

        :param experience_buffer: (np.array)
            Experience buffer in form of [[state][action][reward][td][advantage]]
        :param epochs: Number of epoch training
        """
        observations, actions, rewards, td_diffs, advantages = experience_buffer
        for ep in range(epochs):
            feed_dict = {self.observations_: observations,
                         self.actions_: actions,
                         self.rewards_: rewards,
                         self.td_targets_: td_diffs,
                         self.advantages_: advantages}
            self.sess.run(self.iterator.initializer, feed_dict=feed_dict)

            summary_ = tf.summary.merge([self.var_summary, self.loss_summary, self.grad_summary])
            train_ops = [summary_, self.global_step, self.train_op]

            summary = None
            step = 0
            while True:  # run until batch run out
                try:
                    summary, step, _ = self.sess.run(train_ops)
                except tf.errors.OutOfRangeError:
                    break
        return summary, step

