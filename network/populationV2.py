import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

import numpy as np
import random
from collections import defaultdict 

import utility

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
                 grad_clip_norm  = 0,
                 global_step     = None,
                 initial_step    = 0,
                 trainable       = False,
                 lr_a_gamma      = 1,
                 lr_a_decay_step = 0,
                 entropy_beta    = 0.001,
                 sess            = None,
                 globalAC        = None,
                 num_policy_pool = 10,
                 allowPolicyOverlap = False
                 ):
        
        """ Initialize AC network and required parameters
        
        Keyword arguments:
        in_size - network state input shape
        action_size - action space size (int)
        num_agent - number of agent
        scope - name scope of the network. (special for 'global')
        decay_lr - decay learning rate (boolean)
        lr_actor - learning rate for actor
        grad_clip_norm - normalize gradient clip (0 for no clip)
        global_step - global training step
        initial_step - initial step for local network training
        trainable - true to include training sequence. False otherwise
        lr_a_gamma - decay rate for learning rate (actor)
        lr_a_decay_step - number of step that learning rate decay (actor)
        entropy_beta - entropy weight
        sess - tensorflow session
        globalAC - global network
        num_policy_pool - number of policy population
        """

        # Environment
        self.sess            = sess
        self.globalAC        = globalAC

        # Configurations and Parameters
        self.in_size         = in_size
        self.action_size     = action_size
        self.grad_clip_norm  = grad_clip_norm
        self.scope           = scope
        self.global_step     = global_step
        self.num_policy_pool = num_policy_pool
        self.num_agent       = num_agent
        self.allowPolicyOverlap = allowPolicyOverlap
        self.is_Global  = (scope == 'global')

        with tf.variable_scope(scope), tf.device('/gpu:0'):
            with tf.name_scope('Learning_Rate'):
                self.local_step = tf.Variable(initial_step, trainable=False, name='local_step')
                self.lr_actor = tf.train.exponential_decay(lr_actor,
                                                           self.local_step,
                                                           lr_a_decay_step,
                                                           lr_a_gamma,
                                                           staircase=True,
                                                           name='lr_actor')
            
            ## Learning Rate Variables and Parameters

            ## Optimizer
            self.a_opt_list = [tf.train.AdamOptimizer(self.lr_actor, name='Adam'+str(idx))
                                               for ldx in range(self.num_policy_pool)]

            ## Global Network ##
            # Build actor network weights. (global network does not need training sequence)
            # Create pool of policy to select from
            self.state_input_list = []
            self.actor_list       = []
            self.a_vars_list      = []

            for policyID in range(self.num_policy_pool): # number of policy 
                state_input = tf.placeholder(shape=in_size, dtype=tf.float32, name='state_input_hold'+str(policyID))
                actor, a_vars = self._build_actor_network(state_input, policyID)

                self.state_input_list.append(state_input)
                self.actor_list.append(actor)
                self.a_vars_list.append(a_vars)
                        
            ## Local Network (Trainer)
            if not self.is_Global:
                self.policy_index = self.select_policy(pull=False)
                
                # Loss
                self.action_holder_list      = []
                self.reward_holder_list      = []
                self.actor_loss_list         = []
                self.sample_prob_holder_list = []
                for policyID, actor in enumerate(self.actor_list):
                    with tf.name_scope('Trainer'+str(policyID)):
                        action_holder      = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                        action_OH          = tf.one_hot(action_holder, action_size, name='action_onehot')
                        reward_holder      = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_hold')
                        sample_prob_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_hold')

                        entropy = -tf.reduce_mean(actor * tf.log(actor), name='entropy')
                        obj_func = tf.log(tf.reduce_sum(actor * action_OH, 1)) 
                        exp_v = obj_func * reward_holder * sample_prob_holder + entropy_beta * entropy
                        actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')

                        self.sample_prob_holder_list.append(sample_prob_holder)
                        
                        self.action_holder_list.append(action_holder)
                        self.reward_holder_list.append(reward_holder)
                        self.actor_loss_list.append(actor_loss)

                # Gradient
                self.a_grads_list = []
                for agent_id, (aloss, avar) in enumerate(zip(self.actor_loss_list, self.a_vars_list)):
                    with tf.variable_scope('gradient/agent'+str(agent_id)):
                        a_grads = tf.gradients(aloss, avar)
                    self.a_grads_list.append(a_grads)

                # Sync with Global Network
                self.pull_a_ops = []
                self.update_a_ops = []
                with tf.name_scope('sync'):
                    # Pull global weights to local weights
                    with tf.name_scope('pull'):
                        for lVars, gVars in zip(self.a_vars_list, globalAC.a_vars_list):
                            self.pull_a_ops.append([lVar.assign(gVar) for lVar, gVar in zip(lVars, gVars)])
                    # Push local weights to global weights
                    with tf.name_scope('push'):
                        for opt, lGrads, gVars in zip(self.a_opt_list, self.a_grads_list, globalAC.a_vars_list):
                            self.update_a_ops.append(opt.apply_gradients(zip(lGrads, gVars)))
            
    def _build_actor_network(self, state_input, policyID):
        with tf.variable_scope('actor'+str(policyID)):
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
            actor = layers.fully_connected(actor,
                                           self.action_size,
                                           activation_fn=tf.nn.softmax)
            
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor'+str(policyID))
        return actor, a_vars
    
    ## Pipes
    def update_unitpolicy_global(self, agent_id, state, action, reward, behavior_policy): 
        # Get likelihood of global with states
        policyID = self.policy_index[agent_id]
        feed_dict = {self.globalAC.state_input_list[policyID] : np.stack(state)}
        soft_prob = self.globalAC.sess.run(self.globalAC.actor_list[policyID], feed_dict)
        target_policy = np.array([ar[act] for ar, act in zip(soft_prob,action)])
        
        retraceLambda = 0.202
        sample_weight = []
        running_prob = 1.0
        for idx, (tp, bp) in enumerate(zip(target_policy, behavior_policy)):
            ratio = retraceLambda * min(1.0, tp / bp)
            running_prob *= ratio
            sample_weight.append(running_prob)
        
        # update Sequence
        feed_dict = {
                self.state_input_list[policyID]        : np.stack(state),
                self.action_holder_list[policyID]      : action,
                self.reward_holder_list[policyID]      : reward,
                self.sample_prob_holder_list[policyID] : sample_weight,
                }
        
        aloss = self.sess.run(self.actor_loss_list[policyID], feed_dict)
        ops = self.update_a_ops[policyID]
        self.sess.run(ops, feed_dict)

        return aloss
        
    def pull_global(self):
        ops = [self.pull_a_ops[i] for i in self.policy_index]
        self.sess.run(ops)

    # Return action and critic
    def get_action(self, s):
        a_probs = []
        for aid, policyID in enumerate(self.policy_index):
            feed_dict = {self.state_input_list[policyID] : s[aid:aid+1,]}
            a = self.sess.run(self.actor_list[policyID], feed_dict)
            a_probs.append(a[0])
            
        action_selection = [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]
        
        return action_selection, a_probs
            
    # Policy Random Pool
    def select_policy(self, pull = True):
        assert not self.is_Global
        if self.allowPolicyOverlap:
            policy_index = random.choices(range(self.num_policy_pool), k=self.num_agent)
        else:
            policy_index = random.sample(range(self.num_policy_pool), k=self.num_agent)
            
        policy_index.sort()
        if pull:
            self.pull_global()
        return policy_index
