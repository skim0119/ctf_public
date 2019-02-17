import os
import sys
# os.chdir('./..')
sys.path.insert(0, "./src/")

import configparser

import tensorflow as tf

import time
import numpy as np

import gym_cap.envs.const as CONST


# Data Processing Module
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards
from utility.utils import store_args
from utility.buffer import Trajectory as Replay_buffer

from environment import CtF_Environment as CtF
from environment import CTF_DEFAULT_PARAMS
from network import ActorCritic as Network
from agent import Agent

# the modules that you can use to generate the policy.
import policy.zeros as zeros


# Configuration Parser
config = configparser.ConfigParser()
config.read('config.ini')

BUFFER_SIZE = 1024*8
NUM_AGENT = CONST.NUM_BLUE

GAMMA = config.getfloat('TRAINING', 'DISCOUNT_RATE')

# Save/Summary
LOG_PATH = './logs/run'
MODEL_PATH = './model'
RENDER_PATH = './render'
save_network_frequency = config.getint('TRAINING', 'SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING', 'SAVE_STATISTICS_FREQ')
goal_swap_frequency = 30
ma_step = config.getint('TRAINING', 'MOVING_AVERAGE_SIZE')


class Worker():
    @store_args
    def __init__(self, num_episode, new_network=True, name=None):
        # Initialize Agent 
        self.agent = Agent(CTF_DEFAULT_PARAMS, 1e-4, 1e-4, 'main', LOG_PATH, entropy_beta=0)

        # Initialize Environment
        self.env = CtF(red_policy=zeros)

        # Initialize Buffer
        self.experience_buffer = Replay_buffer(depth=8)
        
        self.progbar = tf.keras.utils.Progbar(num_episode)

    def run(self):
        global_rewards = MA(ma_step)
        global_ep_rewards = MA(ma_step)
        global_length = MA(ma_step)
        global_succeed = MA(ma_step)

        last_none_summary = None
        for episode in range(self.num_episode + 1):
            self.progbar.update(episode)

            # Rollout Episode
            cd_r, r, l, s = self.rollout(episode=episode)
            print(f'\nepisodic reward = {cd_r:.5f}, reward = {r:.5f}, length = {l}, blue_win = {s}\n')
            if len(self.experience_buffer) > BUFFER_SIZE:
                print('Train')
                last_none_summary = self.agent.train(self.experience_buffer)
                self.experience_buffer.clear()

            # Goal Reset
            if episode % goal_swap_frequency == 0 and episode != 0:
                self.agent.set_new_goal()

            # Keep log data
            global_ep_rewards.append(cd_r)
            global_rewards.append(r)
            global_length.append(l)
            global_succeed.append(s)
            if episode % save_stat_frequency == 0 and episode != 0:
                value_dict = {'Records/mean_reward': Worker.global_rewards(),
                              'Records/mean_length': Worker.global_length(),
                              'Records/mean_succeed': Worker.global_succeed(),
                              'Records/mean_episode_reward': Worker.global_ep_rewards()}
                self.agent.record(value_dict, last_none_summary, episode)

            # Save Network
            if episode % save_network_frequency == 0 and episode != 0:
                self.agent.save(MODEL_PATH + '/ctf_policy.ckpt', global_step=episode)

    def rollout(self, episode=0):
        # Initialize run
        trajs = [Replay_buffer(depth=8) for _ in range(NUM_AGENT)]
        self.env.reset()
        s1, g1 = self.env.get_states_and_goals()

        # parameters
        ep_r = 0  # Episodic Reward
        prev_r = 0
        step = 0
        done = False

        # Bootstrap
        a1, v1, gid, g_played = self.agent.get_actions(s1, g1)
        is_alive = self.env.life_status()

        while not done:
            s0, g0 = s1, g1
            a0, v0 = a1, v1
            was_alive = is_alive

            s1, _, env_reward, done, g1 = self.env.step(a0)
            is_alive = self.env.life_status()

            r = env_reward - prev_r - 0.01
            r /= 100.0
            ep_r += r

            if done:
                v1 = [0.0 for _ in range(NUM_AGENT)]
            else:
                a1, v1, gid, g_played = self.agent.get_actions(s1, g1)

            # push to buffer
            for idx in range(NUM_AGENT):
                if was_alive[idx]:
                    trajs[idx].append([s0[idx], a0[idx], r, v0[idx], 0, g0[idx], gid[idx], g_played[idx]])

            # Iteration
            prev_r = env_reward
            step += 1

        # Take discount reward for new trajectory
        # Calculate TD and Advantages
        for idx, traj in enumerate(trajs):
            if len(traj) == 0:
                continue

            rewards = np.array(traj[2])
            values = np.array(traj[3])
            value_ext = np.append(values, [v1[idx]])
            td_target = rewards + GAMMA * value_ext[1:]
            advantages = rewards + GAMMA * value_ext[1:] - value_ext[:-1]
            advantages = discount_rewards(advantages, GAMMA)

            traj[3] = td_target.tolist()
            traj[4] = advantages.tolist()

            self.experience_buffer.extend(traj)

        return ep_r, env_reward, step, self.env.win_state()
