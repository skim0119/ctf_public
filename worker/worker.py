import numpy as np
import random

import tensorflow as tf

import gym
import gym_cap
import gym_cap.envs.const as CONST

# Pre-implemented Policies
import policy.random
import policy.roomba
import policy.policy_RL
import policy.zeros

from utility.utils import discount_rewards, store_args
from utility.dataModule import one_hot_encoder as preproccess

from network.a3c import ActorCritic as Network

from network.base import Tensorboard_utility as TB

# TODO:
# - Try to wait more episode rollout before train.


class Worker(object):
    @store_args
    def __init__(self, name, global_network, sess,
                 global_episodes=None, increment_step_op=None,
                 progbar=None, total_episode=1e5, max_frame=150, vision_range=19,
                 num_blue=4, num_red=4,
                 **kwargs):

        self.env = gym.make("cap-v0").unwrapped
        policy_red = policy.policy_A3C.PolicyGen(color='red')
        self.env.num_blue_ugv = num_blue
        self.env.num_red_ugv = num_red
        self.env.sparse_reward = False
        self.env.reset()
        self.env.red_partial_visibility = False
        self.env.reset(map_size=map_size,
                       policy_red=policy.policy_A3C.PolicyGen(color='red'))

        # Create AC Network for Worker
        self.network = Network(in_size=kwargs['input_shape'],
                               action_size=kwargs['output_size'],
                               scope=name,
                               lr_actor=kwargs['actor_lr'],
                               lr_critic=kwargs['critic_lr'],
                               sess=sess,
                               global_network=global_network)

    def work(self, saver, writer, coord, recorder=None, model_path=None):
        global_rewards = recorder['reward']
        global_length = recorder['length']
        global_succeed = recorder['succeed']

        save_stat_freq = 128
        save_network_freq = 1000

        global_episodes = self.sess.run(self.global_episodes)

        total_step = 0

        with self.sess.as_default(), self.sess.graph.as_default():
            while not coord.should_stop() and global_episodes < self.total_episode:
                _log = global_episodes % save_stat_freq == 0 and global_episodes != 0
                _save = global_episodes % save_network_freq == 0 and global_episodes != 0

                r_episode, length = self.rollout(total_step, _log, writer=writer, episode=global_episodes)
                total_step += length

                global_rewards.append(r_episode)
                global_length.append(length)
                global_succeed.append(self.env.blue_win)
                self.sess.run(self.increment_step_op)
                global_episodes = self.sess.run(self.global_episodes)
                self.progbar.update(global_episodes)

                if _log:
                    summary = tf.Summary()
                    summary.value.add(tag='Records/mean_reward', simple_value=global_rewards())
                    summary.value.add(tag='Records/mean_length', simple_value=global_length())
                    summary.value.add(tag='Records/mean_succeed', simple_value=global_succeed())
                    writer.add_summary(summary, global_episodes)
                    writer.flush()

                # Save network
                if _save:
                    saver.save(self.sess, model_path + '/ctf_policy.ckpt', global_step=global_episodes)

    def rollout(self, total_step, log=False, **kwargs):
        def get_action(self, states):
            feed_dict = {self.netowrk.state_input : states}
            return self.netowrk.run_network(feed_dict)

        s0 = self.env.reset()
        s0 = one_hot_encoder(self.env._env, self.env.get_team_blue, self.vision_range)
        
        # parameters 
        r_episode = 0 
        prev_r = 0

        trajs = [Trajectory(depth=4) for _ in range(self.num_blue)]
        debug_param = []
        
        # Bootstrap
        a1, v1 = self.get_action(s0)
        for step in range(max_ep+1):
            a, v0 = a1, v1
            
            s1, rc, done, info = self.env.step(a)
            s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, self.vision_range)

            r = (rc - prev_r-0.5)

            if step == max_ep and done == False:
                r = -100
                rc = -100
                done = True

            r /= 100.0
            r_episode += r

            if done:
                v1 = [0.0 for _ in range(self.num_blue)]
            else:
                a1, v1 = self.get_action(s1)

            # push to buffer
            for idx in range(len(n_agent)):
                if step == 0 or info['blue_alive'][-2][idx]:
                    trajs[idx].append([s0[idx], a[idx], r, v0[idx]])

            if total_step % update_frequency == 0 or d:
                aloss, closs, entropy = self.train(trajs, v1)
                debug_param.append([aloss, closs, entropy])
                trajs = [Trajectory(depth=4) for _ in range(self.num_blue)]

            # Iteration
            prev_r = rc
            s0 = s1
            total_step += 1

            if done:
                break

        if log:
            writer = kwargs['writer']
            episode = kwargs['episode']
            debug_param = np.sum(debug_param, axis=0)
            summaries = {'summary/entropy' : debug_param[0],
                         'summary/actor_loss' : debug_param[1],
                         'summary/critic_loss' : debug_param[2]}
            for tag, value in summaries.items():
                TB.scalar_logger(tag, value, episode, writer)

        return r_episode, step

    def train(self, trajs, bootstrap=0.0):
        buffer_s, buffer_a, buffer_tdtarget, buffer_adv = [], [], [], []
        for idx, traj in enumerate(trajs):
            if len(traj) == 0:
                continue
            observations = traj[0]
            actions = traj[1]
            rewards = np.array(traj[2])
            values = np.array(traj[3])

            value_ext = np.append(values, [bootstrap[idx]])
            td_target  = rewards + gamma * value_ext[1:]
            advantages = rewards + gamma * value_ext[1:] - value_ext[:-1]
            advantages = discount_rewards(advantages,gamma)

            buffer_s.extend(observations)
            buffer_a.extend(actions)
            buffer_tdtarget.extend(td_target.tolist())
            buffer_adv.extend(advantages.tolist())

        # Update Buffer
        aloss, closs, entropy = self.netowrk.update_global(np.stack(buffer_s),
                buffer_a, buffer_tdtarget, buffer_adv)

        # get global parameters to local ActorCritic 
        self.network.pull_global()

        return aloss, closs, entropy

