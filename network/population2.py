import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random
from collections import defaultdict 

from utility.utils import retrace, retrace_prod

# TODO:
'''
- Instead of limiting number of policy network for local,
    just create full list of network,
- Reduce number of operation node per graph.
'''


# implementation for decentralized action and centralized critic
class Population():
    def __init__(self,
                 in_size,
                 action_size,
                 num_agent,
                 scope,
                 decay_lr        = False,
                 lr_actor        = 1e-4,
                 lr_critic       = 1e-4,
                 grad_clip_norm  = 0,
                 global_step     = None,
                 initial_step    = 0,
                 trainable       = False,
                 lr_a_gamma      = 1,
                 lr_c_gamma      = 1,
                 lr_a_decay_step = 0,
                 lr_c_decay_step = 0,
                 entropy_beta    = 0.001,
                 sess            = None,
                 global_network        = None,
                 num_policy_pool = 10,
                 allow_policy_share= False
                 ):
        
        """ Initialize AC network and required parameters
        
        Keyword arguments:
        in_size - network state input shape
        action_size - action space size (int)
        num_agent - number of agent
        scope - name scope of the network. (special for 'global')
        decay_lr - decay learning rate (boolean)
        lr_actor - learning rate for actor
        lr_critic - learning rate for critic
        grad_clip_norm - normalize gradient clip (0 for no clip)
        global_step - global training step
        initial_step - initial step for local network training
        trainable - true to include training sequence. False otherwise
        lr_a_gamma - decay rate for learning rate (actor)
        lr_c_gamma - decay rate for learning rate (critic)
        lr_a_decay_step - number of step that learning rate decay (actor)
        lr_c_decay_step - number of step that learning rate decay (critic)
        entropy_beta - entropy weight
        sess - tensorflow session
        global_network - global network
        num_policy_pool - number of policy population
        allow_policy_share - if true, allow two agency to have shared policy
        """

        # Environment
        self.sess            = sess
        self.global_network        = global_network

        # Configurations and Parameters
        self.in_size         = in_size
        self.action_size     = action_size
        self.grad_clip_norm  = grad_clip_norm
        self.scope           = scope
        self.global_step     = global_step
        self.num_policy_pool = num_policy_pool
        self.num_agent       = num_agent
        self.allow_policy_share = allow_policy_share
        self.is_global  = (scope == 'global')

        self.retrace_lambda = 0.202

        with tf.variable_scope(scope):
            ## Learning Rate Variables and Parameters
            self.local_step = tf.Variable(initial_step,
                                          trainable=False,
                                          name='local_step')
            self.lr_actor = tf.train.exponential_decay(lr_actor,
                                                       self.local_step,
                                                       lr_a_decay_step,
                                                       lr_a_gamma,
                                                       staircase=True,
                                                       name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic,
                                                        self.local_step,
                                                        lr_c_decay_step,
                                                        lr_c_gamma,
                                                        staircase=True,
                                                        name='lr_critic')
            
        with tf.variable_scope(scope), tf.device('/gpu:0'):
            ## Optimizer
            self.a_opt_list = [tf.train.AdamOptimizer(self.lr_actor) for _ in range(self.num_policy_pool)]
            self.c_opt = tf.train.AdamOptimizer(self.lr_critic)

            ## Global Network ##
            # Build actor network weights. (global network does not need training sequence)
            # Create pool of policy to select from
            self.state_input_list = []
            self.actor_list       = []
            self.a_vars_list      = []

            ## Set policy network
            for policyID in range(self.num_policy_pool): # number of policy 
                state_input_ = tf.placeholder(shape=self.in_size,
                                             dtype=tf.float32,
                                             name='state_input_hold')
                actor, a_vars = self._build_actor_network(state_input_, policyID)

                self.state_input_list.append(state_input)
                self.actor_list.append(actor)
                self.a_vars_list.append(a_vars)
                
            self.critic, self.c_vars, self.critic_state_input = self._build_critic_network()
                        
            ## Local Network (Trainer)
            if not self.is_global:
                self.policy_index = self.select_policy(pull=False) # reset policy index
                _build_actor_loss()
                _build_critic_loss()
                _build_gradient()
                _build_pipeline()

    @Network_Builders
    def _build_actor_network(self, state_input, policy_id):
        scope = 'actor'+str(policy_id)

        with tf.variable_scope(scope):
            net = layers.conv2d(state_input, 32, [5,5], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(layer, [2,2])
            net = layers.conv2d(layer, 64, [3,3], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(layer, [2,2])
            net = layers.conv2d(layer, 64, [2,2], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.flatten(layer)

            net = layers.fully_connected(net, 128)
            net = layers.fully_connected(net, self.action_size,
                                        activation_fn=tf.nn.softmax)
            
        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+scope)

        return actor, vars_list
    
    def _build_critic_network(self):
        """ Common shared critic. Take the observation(state) from all agencies
            First observation state_inputs[0] must be the observation of 'self' agent. """
        state_inputs = [tf.placeholder(shape=self.in_size, dtype=tf.float32, name='cr_state_hold')
                for _ in range(self.num_agent)]
        scope = 'critic'
        with tf.variable_scope(scope):
            nets = []
            for state_input in state_inputs:
                net = layers.conv2d(state_input,
                                    32,
                                    [3,3],
                                    activation_fn=tf.nn.elu,
                                    weights_initializer=layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer(),
                                    padding='SAME')
                net = layers.max_pool2d(layer, [2,2])
                net = layers.conv2d(layer,
                                    64,
                                    [2,2],
                                    activation_fn=tf.nn.elu,
                                    weights_initializer=layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer(),
                                    padding='SAME')
                net = layers.flatten(layer)
                net = layers.fully_connected(net, 128)
                net = layers.fully_connected(net, 1, activation_fn=None)
                nets.append(net)

            net = layers.fully_connected(tf.concat(nets,1), 1, activation_fn=None)
            net = tf.reshape(net, [-1])

        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+scope)

        return net, vars_list, state_inputs

    def _build_actor_loss(self):
        # Placeholders: pipeline for values
        self.action_holder_list  = []
        self.adv_holder_list     = []
        self.actor_loss_list     = []
        self.retrace_holder_list = []

        # Actor Loss
        with tf.name_scope('Actor_Loss'):
            for actor in self.actor_list:
                action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                action_OH = tf.one_hot(action_, self.action_size, name='action_onehot')
                adv_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                retrace_ = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_hold')

                entropy = -tf.reduce_mean(actor * tf.log(actor), name='entropy')
                obj_func = tf.log(tf.reduce_sum(actor * action_OH, 1)) 
                exp_v = obj_func * adv_ * retrace_ + entropy_beta * entropy
                actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')

                self.action_holder_list.append(action_)
                self.adv_holder_list.append(adv_)
                self.actor_loss_list.append(actor_loss)
                self.retrace_holder_list.append(sample_prob_)

    def _build_critic_loss(self):
        ## Make sure critic can get the require result
        with tf.name_scope('Critic_Loss'):
            self.td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
            self.retrace_prod_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_prod_hold')

            td_error = self.td_target_holder - self.critic
            self.critic_loss = tf.reduce_mean(tf.square(td_error)*self.retrace_prod_holder, name='critic_loss')

    def _build_gradient(self):
        # Gradient
        self.a_grads_list = []
        with tf.name_scope('gradient'):
            for agent_id, (aloss, avar) in enumerate(zip(self.actor_loss_list, self.a_vars_list)):
                with tf.variable_scope('agent'+str(agent_id)):
                    a_grads = tf.gradients(aloss, avar)
                self.a_grads_list.append(a_grads)
            self.c_grads = tf.gradients(self.critic_loss, self.c_vars)

    def _build_pipeline(self):
        # Sync with Global Network
        self.pull_a_ops = []
        self.pull_c_ops = []
        self.update_a_ops = []
        self.update_c_ops = []

        # Pull global weights to local weights
        with tf.name_scope('pull'):
            for lVars, gVars in zip(self.a_vars_list, self.global_network.a_vars_list):
                self.pull_a_ops.append([lVar.assign(gVar) for lVar, gVar in zip(lVars, gVars)])
            self.pull_c_ops = [lVar.assign(gVar) for lVar, gVar in zip(self.c_vars, self.global_network.c_vars)]
        # Push local weights to global weights
        with tf.name_scope('push'):
            for opt, lGrads, gVars in zip(self.a_opt_list, self.a_grads_list, self.global_network.a_vars_list):
                self.update_a_ops.append(opt.apply_gradients(zip(lGrads, gVars)))
            self.update_c_ops = self.c_opt.apply_gradient(zip(self.c_grads, self.global_network.c_vars))

    @Manipulation
    def update_full(self, states, actions, advs, td_targets, beta_policies):
        ## Complete update for actor policies and critic
        # All parameters are given for each agents
        a_loss, c_loss = [], []
        for idx in range(self.num_agent):
            s, a, adv, td, beta = states[idx], actions[idx], advs[idx], td_targets[idx], beta_policies[idx]

            # Compute retrace weight
            policy_id = self.policy_index[idx]
            feed_dict = {self.global_network.state_input_list[policy_id] : np.stack(s)}
            soft_prob = self.global_network.sess.run(self.global_network.actor_list[policy_id], feed_dict)
            target_policy = np.array([ar[act] for ar, act in zip(soft_prob,action)])
            retrace_weight = retrace(target_policy, beta_policy, self.retrace_lambda)
            retrace_prod = np.cumprod(retrace_weight)

            # Update specific policy
            feed_dict = {self.state_input_list[policy_id] : np.stack(s),
                         self.action_holder_list[policy_id] : a,
                         self.adv_holder_list[policy_id] : adv,
                         self.retrace_holder_list[policy_id] : retrace_weight}
            loss, _ = self.sess.run([self.actor_loss_list[policy_id], self.update_a_ops[policy_id]], feed_dict)
            a_loss.append(loss)

            # Update critic
            feed_dict = {}
            for idx, s in enumerate(states):
                feed_dict[self.critic_state_input[idx]] = np.stack(s)
            feed_dict[self.critic_state_input[0]], feed_dict[self.critic_state_input[idx]] = 
                feed_dict[self.critic_state_input[idx]], feed_dict[self.critic_state_input[0]] # Swap 0 ~ idx
            feed_dict.update({self.td_target_holder : td,
                              self.retrace_prod_holder : retrace_prod})
            loss, _ = self.sess.run([self.critic_loss, self.update_c_ops], feed_dict)
            c_loss.append(loss)

    def pull_global(self):
        ops = [self.pull_a_ops[i] for i in self.policy_index]
        ops.append(self.pull_c_ops)

        self.sess.run(ops)

    # Return action and critic
    def get_ac(self, states):
        critic_list = []
        a_probs = []
        for agent_id, policy_id in enumerate(self.policy_index):
            s = states[agent_id]
            feed_dict = {self.state_input_list[policy_id] : np.stack(s)}
            for idx, s_ in enumerate(states):
                feed_dict[self.critic_state_input[idx]] = np.stack(s_)
            feed_dict[self.critic_state_input[0]], feed_dict[self.critic_state_input[idx]] = 
                feed_dict[self.critic_state_input[idx]], feed_dict[self.critic_state_input[0]] # Swap 0 ~ idx

            a, c = self.sess.run([self.actor_list[policy_id], self.critic], feed_dict)
            a_probs.append(a[0])
            critic_list.append(c[0])
            
        action_selection = [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]
        
        return action_selection, a_probs, critic_list
            
    # Policy Random Pool
    def select_policy(self, pull = True):
        assert not self.is_global
        if self.allow_policy_share:
            policy_index = random.choices(range(self.num_policy_pool), k=self.num_agent)
        else:
            policy_index = random.sample(range(self.num_policy_pool), k=self.num_agent)
            
        policy_index.sort()
        if pull:
            self.pull_global()
        return policy_index
