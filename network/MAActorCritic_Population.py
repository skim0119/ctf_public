import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random
from collections import defaultdict 

import utility

# TODO:
'''
- Instead of limiting number of policy network for local, just create full list of network,
reduce number of operation node per graph. See if it improves number of operation and time.
'''


# implementation for decentralized action and centralized critic
class MAActorCritic():
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
                 globalAC        = None,
                 num_policy_pool = 10
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
        trainable - true if the network will have training sequence. False otherwise
        lr_a_gamma - decay rate for learning rate (actor)
        lr_c_gamma - decay rate for learning rate (critic)
        lr_a_decay_step - number of step that learning rate decay (actor)
        lr_c_decay_step - number of step that learning rate decay (critic)
        entropy_beta - entropy weight
        sess - tensorflow session
        globalAC - global network
        num_policy_pool - number of policy population
        """

        # Environment
        self.sess                  = sess
        self.globalAC              = globalAC

        # Configurations 
        self.in_size               = in_size
        self.action_size           = action_size
        self.grad_clip_norm        = grad_clip_norm
        self.scope                 = scope
        self.global_step           = global_step
        self.num_policy_pool       = num_policy_pool
        self.num_agent             = num_agent
        
        # Parameters
        self.is_Global             = (scope == 'global')
        self.num_policy = num_agent if self.is_Global else num_policy_pool

        with tf.variable_scope(scope):
            ## Learning Rate Variables and Parameters
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            self.lr_actor = tf.train.exponential_decay(lr_actor, self.local_step,
                                                       lr_a_decay_step, lr_a_gamma, staircase=True, name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic, self.local_step,
                                                       lr_c_decay_step, lr_c_gamma, staircase=True, name='lr_critic')

            ## Optimizer
            #with tf.device('/gpu:0'):
            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

            ## Global Network ##
            # Build actor network weights. (global network does not need training sequence)
            # Create pool of policy to select from
            self.state_input_list = []
            self.actor_list  = []
            self.a_vars_list = []
            self.critic_list      = []
            self.c_vars_list      = []
            for agent_id in range(self.num_policy): # number of policy 
                with tf.variable_scope('agent'+str(agent_id)):
                    state_input = tf.placeholder(shape=in_size, dtype=tf.float32)
                    actor, a_vars, c_layer = self._build_actor_network(state_input, agent_id)
                critic, c_vars = self._build_critic_network(c_layer, agent_id)

                self.state_input_list.append(state_input)
                self.actor_list.append(actor)
                self.a_vars_list.append(a_vars)
                self.critic_list.append(critic)
                self.c_vars_list.append(c_vars)
                        
            ## Local Network (Trainer)
            if not self.is_global:
                self.policy_index = self.select_policy(pull=False)
                
                # Loss
                self.action_holder_list      = []
                self.advantage_holder_list   = []
                self.actor_loss_list         = []
                self.sample_prob_holder_list = []
                self.critic_loss_list      = []
                self.td_target_holder_list = []
                for agent_id in range(self.num_agent):
                    with tf.variable_scope(str(agent_id)+'_loss'):
                        action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name='action_hold')
                        action_OH = tf.one_hot(action_holder, action_size, name='action_onehot')
                        advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                        td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
                        sample_prob_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_hold')

                        actor = self.actor_list[agent_id]
                        entropy = -tf.reduce_mean(actor * tf.log(actor), name='entropy')
                        objective_function = tf.log(tf.reduce_sum(actor * action_OH, 1)) 
                        exp_v = objective_function * advantage_holder * sample_prob_holder + entropy_beta * entropy
                        actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')

                        td_error = td_target_holder - self.critic_list[agent_id]
                        critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

                    self.sample_prob_holder_list.append(sample_prob_holder)
                    self.action_holder_list.append(action_holder)
                    self.advantage_holder_list.append(advantage_holder)
                    self.actor_loss_list.append(actor_loss)

                    self.td_target_holder_list.append(td_target_holder)
                    self.critic_loss_list.append(critic_loss)

                # Gradient
                self.a_grads_list = []
                self.c_grads_list = []
                with tf.name_scope('gradient/'), tf.device('/gpu:0'):
                    for agent_id in range(self.num_agent):
                        with tf.variable_scope('agent'+str(agent_id)):
                            a_grads = tf.gradients(self.actor_loss_list[agent_id], self.a_vars_list[agent_id])
                            c_grads = tf.gradients(self.critic_loss_list[agent_id], self.c_vars_list[agent_id])
                            if grad_clip_norm:
                                a_grads = [(tf.clip_by_norm(grad, grad_clip_norm), var) for grad, var in a_grads if not grad is None]
                                c_grads = [(tf.clip_by_norm(grad, grad_clip_norm), var) for grad, var in c_grads if not grad is None]
                        self.a_grads_list.append(a_grads)
                        self.c_grads_list.append(c_grads)

                # Pipe
                self.pull_a_vars_op    = [ [None for __ in range(self.num_policy_pool)] for _ in range(self.num_agent)]
                self.pull_c_vars_op    = [ [None for __ in range(self.num_policy_pool)] for _ in range(self.num_agent)]

                self.update_a_op       = [ [None for __ in range(self.num_policy_pool)] for _ in range(self.num_agent)]
                self.update_c_op       = [ [None for __ in range(self.num_policy_pool)] for _ in range(self.num_agent)]

                with tf.name_scope('sync'):
                    # Pull global weights to local weights
                    with tf.name_scope('pull'):
                        for aid in range(self.num_agent):
                            for pid in range(self.num_policy_pool):
                                self.pull_a_vars_op[aid][pid] = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars_list[aid], globalAC.a_vars_list[pid])]
                                self.pull_c_vars_op[aid][pid] = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars_list[aid], globalAC.c_vars_list[pid])]
                    # Push local weights to global weights
                    with tf.name_scope('push'):
                        for aid in range(self.num_agent):
                            for pid in range(self.num_policy_pool):
                                self.update_a_op[aid][pid] = self.actor_optimizer.apply_gradients(zip(self.a_grads_list[aid], globalAC.a_vars_list[pid]))
                                self.update_c_op[aid][pid] = self.actor_optimizer.apply_gradients(zip(self.c_grads_list[aid], globalAC.c_vars_list[pid]))
                                
            
    def _build_actor_network(self, state_input, agent_id):
        with tf.variable_scope('actor'):
            layer = slim.conv2d(state_input, 32, [5,5], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID')
            layer = slim.max_pool2d(layer, [2,2])
            layer = slim.conv2d(layer, 64, [3,3], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID')
            layer = slim.max_pool2d(layer, [2,2])
            layer = slim.conv2d(layer, 64, [2,2], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='VALID')
            layer = slim.flatten(layer)

            actor = layers.fully_connected(layer, 128)
            actor = layers.fully_connected(actor, self.action_size,
                                        activation_fn=tf.nn.softmax)
            
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/agent'+str(agent_id)+'/actor')
        return actor, a_vars, layer
    
    def _build_critic_network(self, layer, agent_id):
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE): 
            critic = layers.fully_connected(layer, 1,
                                         activation_fn=None)
            critic = tf.reshape(critic, [-1])

        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')
        return critic, c_vars

    ## Pipes
    def update_unitpolicy_global(self, pid, state, action, advantage, td_target, likelihood_prev): # push
        # Get likelihood of global with states
        gid = self.policy_index[pid]
        feed_dict = {self.globalAC.state_input_list[gid] : np.stack(state)}
        soft_prob = self.globalAC.sess.run(self.globalAC.actor_list[gid], feed_dict)    
        likelihood = np.array([ar[act] for ar, act in zip(soft_prob,action)])
        running_prob = 1.0
        for idx, l in enumerate(likelihood):
            running_prob *= l
            likelihood[idx] = running_prob
        sample_prob = likelihood/likelihood_prev
        
        feed_dict = {
                self.state_input_list[pid]        : np.stack(state),
                self.action_holder_list[pid]      : action,
                self.advantage_holder_list[pid]   : advantage,
                self.td_target_holder_list[pid]   : td_target,
                self.sample_prob_holder_list[pid] : sample_prob
                }
        aloss, closs = self.sess.run([self.actor_loss_list[pid], self.critic_loss_list[pid]], feed_dict)
        
        gid = self.policy_index[pid]
        ops = [self.update_a_op[pid][gid], self.update_c_op[pid][gid]]

        self.sess.run(ops, feed_dict)
        
        return aloss, closs

    def pull_global(self):
        ops = []
        for f, t in enumerate(self.policy_index):
            ops.append(self.pull_a_vars_op[f][t])
            ops.append(self.pull_c_vars_op[f][t])

        self.sess.run(ops)

    # Return critic
    def get_critic(self, s, agent_indices):
        feed_dict = {}
        for idx in range(self.num_agent):
            feed_dict.update( {self.state_input_list[idx] : s[agent_indices==idx]} )
            
        vs = self.sess.run(self.critic_list, feed_dict)
        vs = np.array([v[0] for v in vs])
        return vs

    # Choose Action            
    def get_action(self, s, agent_indices):
        feed_dict = {}
        for idx in range(self.num_agent):
            feed_dict.update( {self.state_input_list[idx] : s[agent_indices==idx] } )

        a_probs = self.sess.run(self.actor_list, feed_dict)    
        a_probs = np.array([ar[0] for ar in a_probs])
        
        action_selection = [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]
        
        return action_selection, a_probs
            
    # Policy Random Pool
    def select_policy(self, pull = True):
        assert not self.is_Global
        policy_index = random.sample(range(self.num_policy_pool), self.num_agent)
        if pull:
            self.pull_global()
        return policy_index
