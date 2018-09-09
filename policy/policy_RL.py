"""Random agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

Last Modified:
    Seung Hyun Kim
    created : Sat Sep  8 19:38:43 2018
"""

import numpy as np
import tensorflow as tp


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

        self.sess = tf.Session()

        model_dir= './../model'
        self.ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver = tf.train.import_meta_graph('ckpt.model_checkpoint_path');
            #saver.restore(sess, tf.train.latest_checkpoint('./'))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Graph is succesfully loaded.')
        else:
            print('Error : Graph is not loaded')
            return

        self.graph = tf.get_default_graph()
        self.state = self.graph.get_tensor_by_name("state:0")
        self.action = self.graph.get_tensor_by_name("action:0")

    def policy(self, agent, obs, agent_id):
        """ Policy
        This method generate an action for given agent.
        Agent is given with limited vision of a field.
        Action is driven from pre-learned network.
        Network will not update for an action.
        """
        # Expand the observation with 3-thickness wall
        # - in order to avoid dealing with the boundary
        obsx, obsy = obs.shape
        _obs = np.ones((obsx+6, obsy+6)) * 8
        _obs[3:3+obsx, 3:3+obsy] = obs
        obs = _obs

        # Initialize Variables
        x, y = agent.get_loc()
        x += 3
        y += 3
        view = obs[x-2:x+3, y-2:y+3] # limited view for the agent

        # Run Graph
        a_dist = self.sess(self.action, feed_dict={self.input:[view]})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)

        return a
        
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

        action_out = []

        for idx, agent in enumerate(agent_list):
            action_out.append(self.policy(agent, observation, idx)
            #action_out.append(self.random.randint(0, 5)) # choose random action
        
        return action_out
