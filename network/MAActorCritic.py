import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random

from network.base import base

import utility

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
        self.td_target_holder_list = []
        self.advantage_holder_list = []
        
        with tf.variable_scope(scope):
            self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
            # Learning Rate Variables
            self.lr_actor = tf.train.exponential_decay(lr_actor, self.local_step,
                                                       lr_a_decay_step, lr_a_gamma, staircase=True, name='lr_actor')
            self.lr_critic = tf.train.exponential_decay(lr_critic, self.local_step,
                                                       lr_c_decay_step, lr_c_gamma, staircase=True, name='lr_critic')

            # Optimizer
            with tf.device('/gpu:0'):
                self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

                # global Network
                # Build actor and critic network weights. (global network does not need training sequence)
                common_layer_list=[]
                for agent_id in range(self.num_agent):
                    # For each agent, build an action policy
                    with tf.name_scope('agent'+str(agent_id)):
                        state_input = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')
                        self.state_input_list.append(state_input)

                        # get the parameters of actor and critic networks
                        actor, common_layer, a_vars = self.build_actor_network(state_input)
                        common_layer_list.append(common_layer)
                
                critic, c_vars = self.build_critic_network(tf.concat(common_layer_list, 1))
                

                        # Local Network
                        # Define how actor policy updates
                        if scope != 'global':
                            action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name='action_holder')
                            action_OH = tf.one_hot(action_holder, action_size)
                            td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                            advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')
                            
                            action_holder_list.append(action_holder)
                            td_target_holder_list.append(td_target_holder)
                            advantage_holder_list.append(advantage_holder)
                            
                            entropy = -tf.reduce_mean(actor * tf.log(actor), name='entropy')
                
                # Define Universal Critic Network
                critic, c_vars = self.build_critic_network(...)
                td_error = 
                
                        
                    # Local Network
                    if scope != 'global':
                        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name='action_holder')
                        self.action_OH = tf.one_hot(self.action_holder, action_size)
                        #self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name='reward_holder')
                        self.td_target_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                        self.advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')


                        self.td_error = self.td_target_holder - self.critic # for gradient calculation (equal to advantages)
                        self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')

                        # Critic (value) Loss
                        with tf.name_scope('critic_train'):
                            self.critic_loss = tf.reduce_mean(tf.square(self.td_error), name='critic_loss') # mse of td error

                        # Actor Loss
                        with tf.name_scope('actor_train'):
                            self.policy_as = tf.reduce_sum(self.actor * self.action_OH, 1) # policy for corresponding state and action
                            self.objective_function = tf.log(self.policy_as) # objective function
                            self.exp_v = self.objective_function * self.advantage_holder + entropy_beta * self.entropy
                            self.actor_loss = tf.reduce_mean(-self.exp_v, name='actor_loss') # or reduce_sum

                        with tf.name_scope('local_grad'):
                            self.a_grads = tf.gradients(self.actor_loss, self.a_vars)
                            self.c_grads = tf.gradients(self.critic_loss, self.c_vars)
                            if self.grad_clip_norm:
                                self.a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.a_grads if not grad is None]
                                self.c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.c_grads if not grad is None]

                        # Sync with Global Network
                        with tf.name_scope('sync'):
                            # Pull global weights to local weights
                            with tf.name_scope('pull'):
                                self.pull_a_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars, globalAC.a_vars)]
                                self.pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, globalAC.c_vars)]

                            # Push local weights to global weights
                            with tf.name_scope('push'):
                                self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_vars))
                                self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_vars))

        if scope != 'global':
            self.build_summarizer()
    def build_actor_network(self, state_input):
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
            
        a_vars = common_vars+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        return actor, commom_layer, a_vars
    
    def build_critic_network(self, layer):
        with tf.variable_scope('critic'):
            critic = layers.fully_connected(layer, 1,
                                         activation_fn=None)
            critic = tf.reshape(critic, [-1])

        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        return critic, c_vars

     # Update global network with local gradients
    def update_global(self, feed_dict):
        al, cl, etrpy, _, __ = self.sess.run([self.actor_loss, self.critic_loss, self.entropy, self.update_a_op, self.update_c_op], feed_dict)
        return al, cl, etrpy
        #_,__,summary_str = self.sess.run([self.update_a_op, self.update_c_op, self.summary_loss], feed_dict)
        #return summary_str

    def pull_global(self):
        self.sess.run([self.pull_a_vars_op, self.pull_c_vars_op])

     # Choose Action
    def choose_action(self, s):
        a_probs = self.sess.run(self.actor, {self.state_input: s})
        
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
        with tf.name_scope('summary'):
            tf.summary.scalar(name='actor_loss', tensor=self.actor_loss)
            tf.summary.scalar(name='critic_loss', tensor=self.critic_loss)
            tf.summary.scalar(name='Entropy', tensor=self.entropy)
        self.summary_loss = tf.summary.merge_all(scope='summary')
        
        with tf.name_scope('Learning_Rate'):
            # Learning Rate
            tf.summary.scalar(name='actor_lr', tensor=self.lr_actor)
            tf.summary.scalar(name='critic_lr', tensor=self.lr_critic)