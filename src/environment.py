import time
import numpy as np

import gym
import gym_cap
import gym_cap.envs.const as CONST

from utility.utils import store_args
from utility.preprocessor import one_hot_encoder_v3 as state_encoder

NUM_CHANNEL = 6
VISION_RANGE = 19
VISION_DX = 2 * VISION_RANGE + 1 
VISION_DY = 2 * VISION_RANGE + 1 
CTF_DEFAULT_PARAMS = {
    'num_channel': NUM_CHANNEL,
    'vision_range': VISION_RANGE,
    'vision_dx': 2 * VISION_RANGE + 1,
    'vision_dy': 2 * VISION_RANGE + 1,
    'input_shape': [None, VISION_DX*VISION_DY*NUM_CHANNEL], 
    'goal_shape': [None, VISION_DX*VISION_DY],
    'action_size': 5,
    'num_agent': CONST.NUM_BLUE
}


class CtF_Environment:
    """ CtF Environment Wrapper for HER+HRL Trial
    """
    @store_args
    def __init__(self,
                 red_policy,
                 map_size=20,
                 vision_range=19,
                 max_frame=150,
                 ):

        # Create CTF Environment and Initialize
        self.env = gym.make("cap-v0").unwrapped
        self.env.reset(map_size=map_size,
                       policy_red=red_policy.PolicyGen(self.env.get_map, self.env.get_team_red))
        self.frame = 0

    # Get state
    def get_states_and_goals(self):
        states, goals = [], []
        for vehicle in self.env.get_team_blue:
            coord = vehicle.get_loc()
            state, goal = state_encoder(state=self.env._env,
                                        coord=coord,
                                        vision_radius=VISION_RANGE, 
                                        normalize_channel=False)
            states.append(state)
            goals.append(goal)
        return np.stack(states), np.stack(goals)

    # Reset simulation
    def reset(self):
        self.env.reset()
        self.frame = 0
        return self.get_states_and_goals()

    # Execute low-level action
    def step(self, actions):
        s, _, done, _ = self.env.step(actions)
        self.frame += 1

        reward = 0
        # Set maximum frame limitation
        if self.frame == self.max_frame and not done:
            reward = -1
            done = True

        # Re-define reward as sparce function
        if self.env.blue_win:
            reward = 1
        elif done:
            reward = -1

        states, goals = self.get_states_and_goals()

        return states, actions, reward, done, goals

    def win_state(self):
        return self.env.blue_win

    def life_status(self):
        return [agent.isAlive for agent in self.env.get_team_blue]
