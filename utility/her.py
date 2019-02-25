import numpy as np
import random

from utility.buffer import Replay_buffer

GOAL_BUFFER_SIZE = 10
# goal-replay will augment the experience.
# Adjust number depend on number of episode per train, number of agent, max episode, and buffer size#
GOAL_REPLAY_SAMPLE = 3

class HER:
    """HER
    
    Hindsight Experience Replay
    Based on https://arxiv.org/abs/1707.01495
    Module does not include full algorithm, but an add-on methods to adopt existing A3C code.

    """

    def __init__(self, depth=6, buffer_size=5000):
        self.replay_buffer = Replay_buffer(depth=depth, buffer_size=buffer_size)
        self.goal_buffer = Replay_buffer(depth=1, buffer_size=GOAL_BUFFER_SIZE)

    def reward(self, s:tuple, a, g:tuple):
        # -[f_g(s)==0]
        # f can be arbitrary. For simplicity, f is identity
        assert len(s) == len(g)
        assert type(s) == type(g) # tuple
        return int(s==g)
        #return -((s==g)==0)

    def action_replay(self, goal):
        """ action replay

        Push new goal into replay buffer
        The final state of the trajectory is set to new sub-goal
        """
        self.goal_buffer.append(goal)

    def sample_goal(self, size=GOAL_REPLAY_SAMPLE):
        if len(self.goal_buffer) <= 0:
            return []
        goal_id = np.random.choice(len(self.goal_buffer), size)
        return [self.goal_buffer[id] for id in goal_id]
        '''
        remainder = max(0,size - len(self.replay_buffer))
        if remainder is not 0:
            return [global_goal]*size
        goal_id = np.random.randint(len(self.replay_buffer), size=size)
        return [self.replay_buffer[id][3] for id in goal_id]
        '''

    def goal_replay(self, s_traj, a_traj, g):
        """ Goal Replay 
    
        Take trajectory, action, and goal, return reward
        """
        reward = []
        for s,a in zip(s_traj, a_traj):
            r = self.reward(s,a,g)
            reward.append(r)
            if r == 1:
                break
        return reward, len(reward)

    def store_transition(self, trajectory:list):
        self.replay_buffer.append(trajectory)

    def sample_minibatch(self, size, shuffle=False):
        if size < len(self.replay_buffer):
            return self.replay_buffer.flush()
        else:
            return self.replay_buffer.pop(size, shuffle)

    def buffer_empty(self):
        return self.replay_buffer.empty()

