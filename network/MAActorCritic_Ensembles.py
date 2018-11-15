import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random
from collections import defaultdict 

from network.base import base

import utility


# implementation for centralized action and critic
class MAActorCritic(base):
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
        in_size         - input size
        action_size     - possible action space (discrete)
        grad_clip_norm  - gradient clip
        scope           - name of the scope. (At least one 'global' network required)
        sess            - tensorflow session# Learning Rate Variables
        globalAC        - use to update or pull network weights from global network
        """

        self.sess=sess
        self.num_policy_pool       = num_policy_pool

        # Parameters
        self.in_size               = in_size
        self.action_size           = action_size
        self.grad_clip_norm        = grad_clip_norm
        self.scope                 = scope
        self.is_Global             = (scope == 'global')
        self.global_step           = global_step
        self.num_agent             = num_agent

        with tf.variable_scope(scope):
            ## Learning Rate Variables and Parameters
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            self.lr_actor = tf.train.exponential_decay(lr_actor, self.local_step,
                                                       lr_a_decay_step, lr_a_gamma, staircase=True, name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic, self.local_step,
                                                       lr_c_decay_step, lr_c_gamma, staircase=True, name='lr_critic')

            ## Optimizer
            with tf.device('/gpu:0'):
                self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

            ## Global Network ##
            # Build actor network weights. (global network does not need training sequence)
            # Create pool of policy to select from
            if self.is_Global:
                self.actor_list  = []
                self.a_vars_list = []
                dummy_input = tf.placeholder(shape-in_size, dtype=tf.float32)
                for agent_id in range(self.num_policy_pool): # number of policy : pool size
                    with tf.variable_scope('agent'+str(agent_id)):
                        actor, a_vars, _ = self.build_actor_network(dummy_input, agent_id)
                    self.actor_list.append(actor)
                    self.a_vars_list.append(a_vars)
            else:
                ## Local Network
                self.state_input_critic = tf.placeholder(shape=in_size,dtype = tf.float32)
                self.policy_index = self.select_policy()

                # Number of policy equals number of agent
                self.state_input_list = []
                self.actor_list       = []
                self.a_vars_list      = []
                self.critic_list      = []
                self.c_vars_list      = []
                for agent_id in range(self.num_agent):
                    # For each agent, build an action policy
                    with tf.variable_scope('agent'+str(agent_id)):
                        state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')
                        # get the parameters of actor and critic networks
                        actor, a_vars, c_layer = self.build_actor_network(state_input, agent_id)
                        
                    critic, c_vars = self.build_critic_network(c_layer, agent_id)

                    self.state_input_list.append(state_input)
                    self.actor_list.append(actor)
                    self.a_vars_list.append(a_vars)
                    self.critic_list.append(critic)
                    self.c_vars_list.append(c_vars)
                        
                ## Local Network Trainer
                # Actor Train
                self.action_holder_list     = []
                self.advantage_holder_list  = []
                self.actor_loss_list        = []
                self.likelihood_holder_list = []
                with tf.name_scope('actor_loss'):
                    for agent_id in range(self.num_agent):
                        with tf.variable_scope('agent'+str(agent_id)):
                            action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name='action_hold')
                            action_OH = tf.one_hot(action_holder, action_size)
                            advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')

                            actor = self.actor_list[agent_id]
                            entropy = -tf.reduce_mean(actor * tf.log(actor))
                            objective_function = tf.log(tf.reduce_sum(actor * action_OH, 1)) 
                            exp_v = objective_function * advantage_holder + entropy_beta * entropy
                            actor_loss = tf.reduce_mean(-exp_v)

                        self.action_holder_list.append(action_holder)
                        self.advantage_holder_list.append(advantage_holder)
                        self.actor_loss_list.append(actor_loss)

                # Critic Train
                self.critic_loss_list      = []
                self.td_target_holder_list = []
                with tf.name_scope('critic_loss'):
                    for agent_id in range(self.num_agent):
                        with tf.variable_scope('agent'+str(agent_id)):
                            td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32)
                            td_error = td_target_holder - self.critic_list[agent_id]
                            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
                            
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

                # Piep
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
                                
                self.pull_global()

            
    def build_actor_network(self, state_input, agent_id):
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
    
    def build_critic_network(self, layer, agent_id):
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE): 
            critic = layers.fully_connected(layer, 1,
                                         activation_fn=None)
            critic = tf.reshape(critic, [-1])

        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')
        return critic, c_vars

    ## Pipes
    def update_unitpolicy_global(self, pid, state, action, advantage, td_target): # push
        feed_dict = {
                self.state_input_list[pid]      : np.stack(state),
                self.action_holder_list[pid]    : action,
                self.advantage_holder_list[pid] : advantage,
                self.td_target_holder_list[pid] : td_target
                }
        aloss, closs = self.sess.run([self.actor_loss_list[pid], self.critic_loss_list[pid]], feed_dict)

        ops_a = []
        ops_c = []
        for f, t in enumerate(self.policy_index):
            ops_a.append(self.update_a_op[f][t])
            ops_c.append(self.update_c_op[f][t])
        self.sess.run(ops_a + ops_c, feed_dict)
        
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
        
        return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]
    
    # Policy Random Pool
    def select_policy(self):
        assert not self.is_Global
        policy_index = random.sample(range(self.num_policy_pool), self.num_agent)
        return policy_index
