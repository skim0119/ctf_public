import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random

from network.base import base

import utility

# implementation for centralized action and critic
class MAActorCritic(base):
    def __init__(self,
                 in_size,
                 action_size,
                 num_agent,
                 scope,
                 decay_lr=False,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm = 0,
                 global_step=None,
                 initial_step=0,
                 trainable = False,
                 lr_a_gamma = 1,
                 lr_c_gamma = 1,
                 lr_a_decay_step = 0,
                 lr_c_decay_step = 0,
                 entropy_beta = 0.001,
                 sess=None,
                 globalAC=None):
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

        # Parameters
        self.in_size = in_size
        self.action_size = action_size
        self.grad_clip_norm = grad_clip_norm
        self.scope = scope
        self.global_step = global_step
        self.num_agent = num_agent
        
        self.state_input_list = []
        self.action_holder_list = []
        self.advantage_holder_list = []

        self.actor_list = []
        self.a_vars_list = []
        self.actor_loss_list = []
        
        with tf.variable_scope(scope):
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            ## Learning Rate Variables
            self.lr_actor = tf.train.exponential_decay(lr_actor, self.local_step,
                                                       lr_a_decay_step, lr_a_gamma, staircase=True, name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic, self.local_step,
                                                       lr_c_decay_step, lr_c_gamma, staircase=True, name='lr_critic')

            ## Optimizer
            with tf.device('/gpu:0'):
                self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

            ## Global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input_critic = tf.placeholder(shape=in_size,dtype=tf.float32)
            for agent_id in range(self.num_agent):
                # For each agent, build an action policy
                with tf.variable_scope('agent'+str(agent_id)):
                    state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')

                    # get the parameters of actor and critic networks
                    actor, a_vars = self.build_actor_network(state_input, agent_id)

                    self.state_input_list.append(state_input)
                    self.actor_list.append(actor)
                    self.a_vars_list.append(a_vars)
            self.critic, self.c_vars = self.build_critic_network(self.state_input_critic)

            if scope == 'global': # Global network only need actor network for weight storage
                return

            ## Training Nodes (for local network)
            
            # Actor Train
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
            with tf.name_scope('critic_loss'):
                self.td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32)
                td_error = self.td_target_holder - self.critic
                self.critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            
            # Gradient
            self.a_grads_list = []
            with tf.name_scope('gradient'), tf.device('/gpu:0'):
                for agent_id in range(self.num_agent):
                    with tf.variable_scope('agent'+str(agent_id)):
                        # Actor Gradient
                        a_grads = tf.gradients(self.actor_loss_list[agent_id], self.a_vars_list[agent_id])
                        if grad_clip_norm:
                            a_grads = [(tf.clip_by_norm(grad, grad_clip_norm), var) for grad, var in a_grads if not grad is None]
                        self.a_grads_list.append(a_grads)
                # Critic Gradient
                c_grads = tf.gradients(self.critic_loss, self.c_vars)
                if grad_clip_norm:
                    c_grads = [(tf.clip_by_norm(grad, grad_clip_norm), var) for grad, var in c_grads if not grad is None]



            # Sync : Pull and Push
            self.pull_a_vars_op = []
            with tf.name_scope('sync'), tf.device('/gpu:0'):
                # Pull global weights to local weights
                with tf.name_scope('pull'):
                    # Actor Weights Pull
                    for a_vars, global_a_vars in zip(self.a_vars_list, globalAC.a_vars_list):
                        self.pull_a_vars_op.extend([local_var.assign(glob_var) for local_var, glob_var in zip(a_vars, global_a_vars)])
                    # Critic Weights Pull
                    self.pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, globalAC.c_vars)]

                # Push local weights to global weights
                self.update_a_op = []
                with tf.name_scope('push'):
                    # Actor Weights Push
                    for a_grads, global_a_vars in zip(self.a_grads_list, globalAC.a_vars_list):
                        self.update_a_op.append(self.actor_optimizer.apply_gradients(zip(a_grads, global_a_vars)))
                    # Critic Weights Push
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(c_grads, globalAC.c_vars))

        if scope != 'global':
            self.build_summarizer()
            
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
        return actor, a_vars
    
    def build_critic_network(self, state_input):
        with tf.variable_scope('critic'):
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
            
            critic = layers.fully_connected(layer, 1,
                                         activation_fn=None)
            critic = tf.reshape(critic, [-1])

        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic/')
        return critic, c_vars

    # Pipes
    def update_unitpolicy_global(self, pid, state, action, advantage, td_target, summarize=False):
        feed_dict = {
                self.state_input_list[pid]      : np.stack(state),
                self.action_holder_list[pid]    : action,
                self.advantage_holder_list[pid] : advantage,
                self.td_target_holder           : td_target,
                self.state_input_critic         : np.stack(state)
                }
        aloss, closs, __, ___ = self.sess.run([self.actor_loss_list[pid], self.critic_loss, self.update_a_op[pid], self.update_c_op], feed_dict)
        
        return aloss, closs

    def pull_global(self):
        self.sess.run([self.pull_a_vars_op, self.pull_c_vars_op])

    # Return critic
    def get_critic(self, s, agent_indices):
        feed_dict = {self.state_input_critic : s}
        vs = self.sess.run(self.critic, feed_dict)
        return vs

     # Choose Action
    def get_action(self, s, agent_indices):
        feed_dict = {}
        for idx in range(self.num_agent):
            feed_dict.update( {self.state_input_list[idx] : s[agent_indices==idx] } )

        a_probs = self.sess.run(self.actor_list, feed_dict)    
        a_probs = np.array([ar[0] for ar in a_probs])
        
        return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]
    
    def build_summarizer(self):
        # Summary
        # Histogram output
        '''with tf.name_scope('debug_parameters'):
            tf.summary.histogram('output', self.actor)
            tf.summary.histogram('critic', self.critic)        
            tf.summary.histogram('action', self.action_holder)
            tf.summary.histogram('objective_function', self.objective_function)
            tf.summary.histogram('td_target', self.td_target_holder)
            tf.summary.histogram('adv_in', self.advantage_holder)
            #tf.summary.histogram('rewards_in', self.reward_holder)'''
        
        # Graph summary Loss
        self.summary_loss_list = []
        with tf.name_scope('summary/'):
            cls = tf.summary.scalar(name='critic_loss', tensor=self.critic_loss)
            for agent_id in range(self.num_agent):
                als = tf.summary.scalar(name='actor_loss', tensor=self.actor_loss_list[agent_id])
                self.summary_loss_list.append( tf.summary.merge([als, cls]) )
            #tf.summary.scalar(name='Entropy', tensor=self.entropy)
