import os
# os.chdir('./..')

import configparser

import tensorflow as tf
import matplotlib.pyplot as plt

import time
import gym, gym_cap
import gym_cap.envs.const as CONST
import numpy as np
import random
import math

# the modules that you can use to generate the policy. 
import policy.zeros

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards
from utility.buffer import Trajectory as Replay_buffer

from network.network import ActorCritic as Network

import imageio



class Worker(Object):
    """ Worker """

    # Configuration Parser
    config = configparser.ConfigParser()
    config.read(config_path)

    # Default configuration constants
    N_CHANNEL = 6
    VISION_RANGE = 19 # CNN Size
    VISION_DX, VISION_DY = 2*VISION_RANGE+1, 2*VISION_RANGE+1
    INPUT_SHAPE = [None,VISION_DX*VISION_DY*N_CHANNEL]
    ACTION_SHAPE = config.getint('DEFAULT','ACTION_SPACE')

    BATCH_SIZE = 32
    NUM_AGENT = CONST.NUM_BLUE
    NUM_RED = CONST.NUM_RED

    LR_A = 1e-4
    LR_C = 2e-4
    GAMMA = config.getfloat('TRAINING', 'DISCOUNT_RATE')

    MAP_SIZE = 20
    MAX_EP = config.getint('TRAINING','MAX_STEP')

    # Containers for Statistics
    ma_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')
    global_rewards = MA(ma_step)
    global_ep_rewards = MA(ma_step)
    global_length = MA(ma_step)
    global_succeed = MA(ma_step)

    ## Save/Summary
    save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
    save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ')


    def __init__(self, episode_num, name=None, log_path=None):
        # Initialize TF Session
        network = Network(input_shape=INPUT_SHAPE,
                          action_shape=ACTION_SPACE,
                          lr_actor=LR_A,
                          lr_critic=LR_C,
                          entropy_beta=0.05,
                          name=name)
                
        self.writer = tf.summary.FileWriter(write_logpath, network.graph)

        # Initialize Environment
        self.env = gym.make("cap-v0").unwrapped
        self.env.reset(map_size=MAP_SIZE,
                       policy_red=policy.zeros.PolicyGen(self.env.get_map, self.env.get_team_red))

        self.episode_num = episode_num
        self.log_path = log_path

    @staticmethod
    def run(episodes=100000):
        global global_rewards, global_ep_rewards, global_length, global_succeed
        batch_count = 0
        for episode in tqdm(range(total_episodes + 1)):
            ep_r, r, length, batch_count, s, summary_ = rollout(init_step=batch_count,
                                                                episode=episode)

            global_ep_rewards.append(ep_r)
            global_rewards.append(r)
            global_length.append(length)
            global_succeed.append(s)

            #progbar.update(episode)

            if summary_ != None or (episode % save_stat_frequency == 0 and episode != 0):
                summary = tf.Summary()
                summary.value.add(tag='Records/mean_reward', simple_value=global_rewards())
                summary.value.add(tag='Records/mean_length', simple_value=global_length())
                summary.value.add(tag='Records/mean_succeed', simple_value=global_succeed())
                summary.value.add(tag='Records/mean_episode_reward', simple_value=global_ep_rewards())
                writer.add_summary(summary,episode)
                if summary_ is not None:
                    writer.add_summary(summary_,episode)
                writer.flush()

            if episode % save_network_frequency == 0 and episode != 0:
                saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=episode)


#progbar = tf.keras.utils.Progbar(total_episodes,interval=1)



def get_action(states):
    """Run graph to get action for each agents"""
    actions, values = network.feed_forward(states)

    return actions, values

def rollout(init_step=0, episode=0, train=True):
    # Initialize run
    batch_count = init_step
    experience = []

    s0 = env.reset()
    s0 = one_hot_encoder(env._env, env.get_team_blue, vision_range, flatten=True)
    # parameters
    ep_r = 0 # Episodic Reward
    prev_r = 0
    step = 0
    d = False

    # Trajectory Buffers
    trajs = [Trajectory(depth=4) for _ in range(n_agent)]

    # RNN Initializer
    rnn_states = [sess.run(network.rnn_eval_init)
                      for _ in range(n_agent)]

    # Bootstrap
    a1, v1, final_states = get_action(s0, rnn_states)
    is_alive = [ag.isAlive for ag in env.get_team_blue]
    buffer_d = []

    while step <= max_ep and not d:
        a, v0 = a1, v1
        was_alive = is_alive
        rnn_states = final_states

        s1, rc, d, _ = env.step(a)
        s1 = one_hot_encoder(env._env, env.get_team_blue, vision_range, flatten=True)

        is_alive = [ag.isAlive for ag in env.get_team_blue]
        r = rc - prev_r - 0.01

        if step == max_ep and d == False:
            r = -100
            rc = -100
            d = True

        r /= 100.0
        ep_r += r

        if d:
            v1 = [0.0 for _ in range(n_agent)]
        else:
            a1, v1, final_states = get_action(s1, rnn_states)

        # push to buffer
        buffer_d.append(d)
        for idx, agent in enumerate(env.get_team_blue):
            if was_alive[idx]:
                trajs[idx].append([s0[idx],
                                   a[idx],
                                   r,
                                   v0[idx],
                                  ])

        # Iteration
        prev_r = rc
        step += 1
        s0 = s1

    if not train:
        return

    # Normalise rewards
    ds = np.array(buffer_d)
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue

        # Discount Reward
        _ds = ds[:len(traj)]
        _rew = np.array(traj[2])
        _rew = discount_rewards(_rew, 0.98)
        #_rew = np.clip(_rew / reward_statistics.std, -10, 10)
        _base = np.array(traj[3])  # Bootstrap

        bs, ba, br, bbas = np.stack(traj[0]), np.array(traj[1]), np.array(_rew), np.array(_base)
        #np.reshape(traj[0], [len(traj[0])] + in_size[-3:]),

        experience.append([bs, ba, br, bbas])
        batch_count += 1

    # Update ppo
    if batch_count >= batch_size:
        #reward_statistics.update(np.array(np.concatenate(list(zip(*experience))[2])))

        # print(f'experience length: {len(experience)}')
        graph_summary, global_step = network.feed_backward(experience)

        # Reset
        batch_count, experience = 0, []
    else:
        graph_summary = None

    return ep_r, rc, step,batch_count, env.blue_win, graph_summary


