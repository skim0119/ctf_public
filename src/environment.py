import time
import numpy as np

import gym
import gym_cap
import gym_cap.envs.const as CONST

from utility.utils import store_args


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
    def get_state(self):
        return self.env._env

    # Reset simulation
    def reset(self):
        self.env.reset()
        self.frame = 0
        return self.get_state()

    # Execute low-level action
    def step(self, action):
        s, _, done, _ = self.env.step(action)
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

        return self.get_state(), reward, done

    def win_state(self):
        return self.env.blue_win
