import os
import sys
# os.chdir('./..')
sys.path.insert(0, "./src/")

import configparser

import tensorflow as tf
import matplotlib.pyplot as plt

import time
import gym, gym_cap
import gym_cap.envs.const as CONST
import numpy as np
import random
import math

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards
from utility.buffer import Trajectory as Replay_buffer

# the modules that you can use to generate the policy. 
import policy.zeros as zeros

from network import ActorCritic as Network

import imageio


# Configuration Parser
config = configparser.ConfigParser()
config.read('config.ini')

# Default configuration constants
N_CHANNEL = 6
VISION_RANGE = 19 # CNN Size
VISION_DX, VISION_DY = 2*VISION_RANGE+1, 2*VISION_RANGE+1
INPUT_SHAPE = [None,VISION_DX,VISION_DY,N_CHANNEL]
ACTION_SHAPE = config.getint('DEFAULT','ACTION_SPACE')

BATCH_SIZE = 8192
NUM_AGENT = CONST.NUM_BLUE
NUM_RED = CONST.NUM_RED

LR_A = 1e-4
LR_C = 2e-4
GAMMA = config.getfloat('TRAINING', 'DISCOUNT_RATE')

MAP_SIZE = 20
MAX_EP = config.getint('TRAINING','MAX_STEP')

# Containers for Statistics
ma_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')

## Save/Summary
LOG_PATH = './logs/run'
MODEL_PATH = './model'
RENDER_PATH = './render'
save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ')


class Worker():
    """ Worker """
    global_rewards = MA(ma_step)
    global_ep_rewards = MA(ma_step)
    global_length = MA(ma_step)
    global_succeed = MA(ma_step)

    global_episode = None

    def __init__(self, episode_num, name=None):
        # Initialize TF Session
        self.network = Network(input_shape=INPUT_SHAPE,
                               output_shape=ACTION_SHAPE,
                               lr_actor=LR_A,
                               lr_critic=LR_C,
                               entropy_beta=0.05,
                               name=name)
        Worker.global_episode = self.network.global_step
                
        self.writer = tf.summary.FileWriter(LOG_PATH, self.network.graph)

        # Initialize Environment
        self.env = gym.make("cap-v0").unwrapped
        self.env.reset(map_size=MAP_SIZE,
                       policy_red=zeros.PolicyGen(self.env.get_map, self.env.get_team_red))

        self.episode_num = episode_num
        self.experience_buffer = Replay_buffer(depth=5)
        self.progbar = tf.keras.utils.Progbar(episode_num)

    def get_action(self, states):
        """Run graph to get action for each agents"""
        actions, values = self.network.feed_forward(states)

        return actions, values

    def run(self):
        batch_count = 0
        for episode in range(self.episode_num + 1):
            self.progbar.update(episode)
            cd_r, r, l, s, summary_ = self.rollout(episode=episode)

            Worker.global_ep_rewards.append(cd_r)
            Worker.global_rewards.append(r)
            Worker.global_length.append(l)
            Worker.global_succeed.append(s)

            if summary_ != None or (episode % save_stat_frequency == 0 and episode != 0):
                summary = tf.Summary()
                summary.value.add(tag='Records/mean_reward', simple_value=Worker.global_rewards())
                summary.value.add(tag='Records/mean_length', simple_value=Worker.global_length())
                summary.value.add(tag='Records/mean_succeed', simple_value=Worker.global_succeed())
                summary.value.add(tag='Records/mean_episode_reward', simple_value=Worker.global_ep_rewards())
                self.writer.add_summary(summary, episode)
                if summary_ is not None:
                    self.writer.add_summary(summary_,episode)
                self.writer.flush()

            if episode % save_network_frequency == 0 and episode != 0:
                self.network.save(MODEL_PATH+'/ctf_policy.ckpt', global_step=Worker.global_step)

    def rollout(self, episode=0, train=True):
        # Initialize run
        trajs = [Replay_buffer(depth=5) for _ in range(NUM_AGENT)]

        s0 = self.env.reset()
        s0 = one_hot_encoder(self.env._env, self.env.get_team_blue, VISION_RANGE)
        # parameters
        ep_r = 0 # Episodic Reward
        prev_r = 0
        step = 0
        d = False

        # Bootstrap
        a1, v1 = self.get_action(s0)
        is_alive = [ag.isAlive for ag in self.env.get_team_blue]

        while step <= MAX_EP and not d:
            a, v0 = a1, v1
            was_alive = is_alive

            s1, env_reward, d, _ = self.env.step(a.tolist())
            s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, VISION_RANGE)
            is_alive = [ag.isAlive for ag in self.env.get_team_blue]
            
            r = env_reward - prev_r - 0.01

            if step == MAX_EP and d == False:
                r = -100
                d = True

            r /= 100.0
            ep_r += r

            if d:
                v1 = [0.0 for _ in range(NUM_AGENT)]
            else:
                a1, v1 = self.get_action(s1)

            # push to buffer
            for idx, agent in enumerate(self.env.get_team_blue):
                if was_alive[idx]:
                    trajs[idx].append([s0[idx],
                                       a[idx],
                                       r,
                                       v0[idx],
                                       0
                                       ])

            # Iteration
            prev_r = env_reward
            step += 1
            s0 = s1

        graph_summary, global_step = None, 0
        if train:
            # Discount/Normalize rewards for each trajs
            for idx, traj in enumerate(trajs):
                if len(traj) == 0:
                    continue

                rewards = np.array(traj[2])
                values = np.array(traj[3])
                value_ext = np.append(values, [v1[idx]])
                td_target  = rewards + GAMMA * value_ext[1:]
                advantages = rewards + GAMMA * value_ext[1:] - value_ext[:-1]
                advantages = discount_rewards(advantages,GAMMA)

                traj[3] = td_target.tolist()
                traj[4] = advantages.tolist()

                self.experience_buffer.extend(traj)

            # Update ppo
            if len(self.experience_buffer) >= BATCH_SIZE:
                stime = time.time()
                graph_summary, global_step = self.network.feed_backward(self.experience_buffer.nparray(), epochs=5)
                self.experience_buffer.clear()
                etime = time.time()
                print(f'\nTraining Duration: {etime-stime} sec')

        return ep_r, env_reward, step, self.env.blue_win, graph_summary

