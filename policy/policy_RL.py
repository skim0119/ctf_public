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
import gym_cap.envs.const as CONST

UNKNOWN  = CONST.UNKNOWN # -1
TEAM1_BG = CONST.TEAM1_BACKGROUND # 0
TEAM2_BG = CONST.TEAM2_BACKGROUND # 1
TEAM1_AG = CONST.TEAM1_UGV # 2
TEAM2_AG = CONST.TEAM2_UGV # 4
TEAM1_FL = CONST.TEAM1_FLAG # 6
TEAM2_FL = CONST.TEAM2_FLAG # 7
OBSTACLE = CONST.OBSTACLE # 8
DEAD     = CONST.DEAD # 9
SELECTED = CONST.SELECTED # 10
COMPLETED= CONST.COMPLETED # 11

VISION_RANGE = 10 #CONST.UGV_RANGE
VISION_dX    = 2*VISION_RANGE+1
VISION_dY    = 2*VISION_RANGE+1


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

            self.graph = tf.get_default_graph()
            self.state = self.graph.get_tensor_by_name("state:0")
            self.action = self.graph.get_tensor_by_name("action:0")
            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            print('Error : Graph is not loaded')
            return

    def one_hot_encoder(self, state, agents):
        ret = np.zeros((len(agents),VISION_dX,VISION_dY,6))
        # team 1 : (1), team 2 : (-1), map elements: (0)
        map_channel = {UNKNOWN:0, DEAD:0,
                       TEAM1_BG:1, TEAM2_BG:1,
                       TEAM1_AG:2, TEAM2_AG:2,
                       3:3, 5:3, # UAV, does not need to be included for now
                       TEAM1_FL:4, TEAM2_FL:4,
                       OBSTACLE:5}
        map_color   = {UNKNOWN:1, DEAD:0, OBSTACLE:1,
                       TEAM1_BG:1, TEAM2_BG:-1,
                       TEAM1_AG:1, TEAM2_AG:-1,
                       3:1, 5:-1, # UAV, does not need to be included for now
                       TEAM1_FL:1, TEAM2_FL:-1}
        
        # Expand the observation with 3-thickness wall
        # - in order to avoid dealing with the boundary
        sx, sy = state.shape
        _state = np.ones((sx+2*VISION_RANGE, sy+2*VISION_RANGE)) * OBSTACLE # 8 for obstacle
        _state[VISION_RANGE:VISION_RANGE+sx, VISION_RANGE:VISION_RANGE+sy] = state
        state = _state

        for idx,agent in enumerate(agents):
            # Initialize Variables
            x, y = agent.get_loc()
            x += VISION_RANGE
            y += VISION_RANGE
            vision = state[x-VISION_RANGE:x+VISION_RANGE+1,y-VISION_RANGE:y+VISION_RANGE+1] # extract the limited view for the agent (5x5)
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
        action_prob = self.sess.run(self.action, feed_dict={self.state:view}) # Action Probability
        action_out = [np.random.choice(5, p=action_prob[x]/sum(action_prob[x])) for x in range(len(agent_list))]

        return action_out
