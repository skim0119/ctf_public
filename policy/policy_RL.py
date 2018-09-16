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
import tensorflow as tf


class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self, free_map, agent_list, sess):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """

        #self.sess = tf.Session()
        self.sess = sess

        model_dir= './model'
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta');
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            print('Error : Graph is not loaded')
            return

        self.graph = tf.get_default_graph()
        self.state = self.graph.get_tensor_by_name("state:0")
        self.action = self.graph.get_tensor_by_name("action:0")

    def one_hot_encoder(self, state, agents):
        ret = np.zeros((len(agents),7,7,6))
        # team 1 : (1), team 2 : (-1)
        map_channel = {-1:0, 0:1, 1:1, 2:2, 4:2, 3:3, 5:3, 6:4, 7:4, 8:5, 9:0}
        map_color   = {-1:0, 0:1, 2:1, 3:1, 6:1, 1:-1, 4:-1, 5:-1, 7:-1, 8:1, 9:0}
        #reorder = {0:0, 1:1, 2:2, 4:3, 6:4, 7:5, 8:6, 9:7} # CHANGE

        # Expand the observation with 3-thickness wall
        # - in order to avoid dealing with the boundary
        sx, sy = state.shape
        _state = np.ones((sx+8, sy+8)) * 8 # 8 for obstacle
        _state[4:4+sx, 4:4+sy] = state
        state = _state

        for idx,agent in enumerate(agents):
            # Initialize Variables
            x, y = agent.get_loc()
            x += 4
            y += 4
            vision = state[x-3:x+4, y-3:y+4] # limited view for the agent (5x5)
            for i in range(len(vision)):
                for j in range(len(vision[0])):
                    if vision[i][j] != -1:
                        channel = map_channel[vision[i][j]]
                        ret[idx][i][j][channel] = map_color[vision[i][j]]
        return ret
        
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

        view = self.one_hot_encoder(observation, agent_list)
        ap = self.sess.run(self.action, feed_dict={self.state:view})
        action_out = [np.random.choice(5, p=ap[x]/sum(ap[x])) for x in range(len(agent_list))]

        return action_out
