import numpy as np

import tensorflow as tf

import gym

# Pre-implemented Policies
import policy.roomba
import policy.policy_A3C

from utility.utils import discount_rewards, store_args
from utility.buffer import Trajectory
from utility.dataModule import one_hot_encoder as preprocess

from network.a3c import ActorCritic as Network

from network.base import Tensorboard_utility as TB

# TODO:
# - Re-organize hyperparameters
#   - Fix out-source keyward parameters
# - Try full trace of the session run


class Run_Param:
    """ Run-time Parameter Module
    All parameter included in this modules are required parameters for Worker class
    Additional arguments can be added for later purpose
    All the parameters provided by keyword arguments are stored also.
    To override the parameter, the name of the variable must match the keyword.
    """
    # Training Configuration
    total_episode = 2e5
    max_frames = 150
    update_frequency = 32
    selfplay_update_frequency = 1000

    # Initialize Iteration Counters
    total_step = 0

    # Network Configuration
    vision_range = 19

    # Environment Configuration
    num_blue = 5
    num_red = 5
    map_size = 20

    # Summary / Save Parameter
    log_frequency = 128
    save_frequency = 1000

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Worker(object, Run_Param):
    @store_args
    def __init__(
            self, name, global_network, sess,
            global_episodes=None, increment_step_op=None,
            progbar=None,
            **kwargs
    ):
        Run_Param.__init__(self, kwargs)

        # Create Environment
        self.env = gym.make("cap-v0").unwrapped
        self.env.num_blue_ugv = self.num_blue
        self.env.num_red_ugv = self.num_red
        self.env.sparse_reward = False
        self.env.reset()
        self.env.red_partial_visibility = False
        self.env.reset(
            map_size=self.map_size,
            policy_red=policy.roomba.PolicyGen(
                self.env.get_map,
                self.env.get_team_red
            )
        )
        self.env()

        self.policy_red = policy.policy_A3C.PolicyGen(color='red')

        # Create AC Network for Worker
        self.network = Network(
            in_size=kwargs['input_shape'],
            action_size=kwargs['output_size'],
            scope=name,
            lr_actor=kwargs['actor_lr'],
            lr_critic=kwargs['critic_lr'],
            sess=sess,
            global_network=global_network
        )

    def work(self, saver, writer, coord, recorder=None, model_path=None):
        global_rewards = recorder['reward']
        global_length = recorder['length']
        global_succeed = recorder['succeed']

        global_episodes = self.sess.run(self.global_episodes)

        with self.sess.as_default(), self.sess.graph.as_default():
            while not coord.should_stop() and global_episodes < self.total_episode:
                _log = global_episodes % self.log_frequency == 0 and global_episodes != 0
                _save = global_episodes % self.save_frequency == 0 and global_episodes != 0
                _selfplay_update = global_episodes % self.selfplay_update_frequency == 0 and global_episodes > self.save_frequency

                r_episode, length = self.rollout(_log, writer=writer, episode=global_episodes)

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

                if _selfplay_update:
                    self.policy_red.reset_network_weight()

    def rollout(self, log=False, **kwargs):
        def get_action(states):
            return self.network.run_network(states)

        if kwargs['episode'] > 75000:
            policy_red = self.policy_red
        else:
            policy_red = policy.roomba.PolicyGen(self.env.get_map, self.env.get_team_red)

        s0 = self.env.reset(policy_red=policy_red)
        s0 = preprocess(self.env._env, self.env.get_team_blue, self.vision_range)

        # parameters
        r_episode = 0
        prev_r = 0

        trajs = [Trajectory(depth=4) for _ in range(self.num_blue)]
        debug_param = []

        # Bootstrap
        a1, v1 = get_action(s0)
        for step in range(self.max_frame + 1):
            a, v0 = a1, v1

            s1, rc, done, info = self.env.step(a)
            s1 = preprocess(self.env._env, self.env.get_team_blue, self.vision_range)

            r = (rc - prev_r - 0.5)

            if step == self.max_frame and not done:
                r = -100
                rc = -100
                done = True

            r /= 100.0
            r_episode += r

            if done:
                v1 = [0.0 for _ in range(self.num_blue)]
            else:
                a1, v1 = get_action(s1)

            # push to buffer
            for idx in range(self.num_blue):
                if step == 0 or info['blue_alive'][-2][idx]:
                    trajs[idx].append([s0[idx], a[idx], r, v0[idx]])

            if self.total_step % self.update_frequency == 0 or done:
                aloss, closs, entropy = self.train(trajs, v1)
                debug_param.append([aloss, closs, entropy])
                trajs = [Trajectory(depth=4) for _ in range(self.num_blue)]

            # Iteration
            prev_r = rc
            s0 = s1
            self.total_step += 1

            if done:
                break

        if log:
            writer = kwargs['writer']
            episode = kwargs['episode']
            debug_param = np.mean(debug_param, axis=0)
            summaries = {
                'summary/Entropy': debug_param[2],
                'summary/actor_loss': debug_param[0],
                'summary/critic_loss': debug_param[1]
            }
            for tag, value in summaries.items():
                TB.scalar_logger(tag, value, episode, writer)

        return r_episode, step

    def train(self, trajs, bootstrap=0.0):
        gamma = 0.98
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
            advantages = discount_rewards(advantages, gamma)

            buffer_s.extend(observations)
            buffer_a.extend(actions)
            buffer_tdtarget.extend(td_target.tolist())
            buffer_adv.extend(advantages.tolist())

        # Update Buffer
        aloss, closs, entropy = self.network.update_global(
            np.stack(buffer_s),
            buffer_a,
            buffer_tdtarget,
            buffer_adv
        )

        # get global parameters to local ActorCritic
        self.network.pull_global()

        return aloss, closs, entropy
