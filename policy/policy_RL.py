"""Random agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

Last Modified:
    Seung Hyun Kim
    created : Sat Sep  30 2018
"""

import numpy as np
import tensorflow as tf
from DataModule import one_hot_encoder

class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self, free_map, agent_list):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """
        self.model_dir= './model/NO_RED_04_EXPBF_only'
        self.sess = tf.Session()
        #self.sess = sess

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta');
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            self.graph = tf.get_default_graph()
            self.state = self.graph.get_tensor_by_name("state:0")
            self.action = self.graph.get_tensor_by_name("action:0")
            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            raise NameError
            print('Error : Graph is not loaded')

    def gen_action(self, agent_list, observation, free_map=None):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).
            
        Returns:
            action_out (list): list of integers as actions selected for team.

        Note:
            The graph is not updated in this session. It only returns action for given input.
        """

        view = one_hot_encoder(observation, agent_list)
        action_prob = self.sess.run(self.action, feed_dict={self.state:view}) # Action Probability
        action_out = [np.random.choice(5, p=action_prob[x]/sum(action_prob[x])) for x in range(len(agent_list))]

        return action_out
