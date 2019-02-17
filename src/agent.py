import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from network import ActorCritic as Network

from utility.utils import store_args, discount_rewards
from utility.preprocessor import one_hot_encoder_v3 as state_encoder


class Agent():
    @store_args
    def __init__(self, ctf_params, lr_actor, lr_critic, name, log_path,
                 entropy_beta=0.05, new_network=True):
        # Launch TensorFlow Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                    allow_growth=True)
        session_config = tf.ConfigProto(gpu_options=gpu_options)
        # log_device_placement=True)
        self.sess = tf.Session(config=session_config)
        for device in self.sess.list_devices():  # Verbose: list all device
            print(device)

        # Define Hierarchy of Network
        # - It is double layer, but it can be multi-layer
        self.meta_controller = Network(input_shape=ctf_params['input_shape'],
                                       goal_shape=ctf_params['goal_shape'],
                                       output_size=5,
                                       sess = self.sess,
                                       lr_actor=lr_actor,
                                       lr_critic=lr_critic,
                                       scope='Meta_Graph')
        self.controller = Network(input_shape=ctf_params['input_shape'],
                                  goal_shape=ctf_params['goal_shape'],
                                  output_size=ctf_params['action_size'],
                                  sess = self.sess,
                                  lr_actor=lr_actor,
                                  lr_critic=lr_critic,
                                  scope='Sub_Graph')
        
        # Saver and Restorer
        self.writer = tf.summary.FileWriter(log_path)
        self.saver = tf.train.Saver(max_to_keep=3)

        ckpt = tf.train.get_checkpoint_state('./model')
        if new_network:
            self.sess.run(tf.global_variables_initializer())
            print("Initialized Variables")
        elif ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load Model : ", ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("Initialized Variables")

        # Hand-crafted goals (First goal is given from environment)
        vision_dx = ctf_params['vision_dx']
        vision_dy = ctf_params['vision_dy']
        self.goals = np.random.randint(5,size=ctf_params['num_agent'])
        self.h_goals = np.zeros((4, vision_dx, vision_dy))
        self.h_goals[0,vision_dx//2-3, vision_dy//2] = 1
        self.h_goals[0,vision_dx//2+3, vision_dy//2] = 1
        self.h_goals[0,vision_dx//2, vision_dy//2-3] = 1
        self.h_goals[0,vision_dx//2, vision_dy//2+3] = 1
        self.h_goals = np.reshape(self.h_goals, (4, -1))

    def save(self, path, global_step=None):
        assert global_step is not None
        self.saver.save(self.sess, save_path=path, global_step=global_step)

    def set_new_goal(self):
        self.goals = np.random.randint(5,size=ctf_params['num_agent'])

    def get_actions(self, states, env_goals):
        actions, values = [], []
        goals = []
        for idx, goal_index in enumerate(self.goals):
            if goal_index == 0:
                goal = env_goals[idx]
            else:
                goal = self.h_goals[goal_index-1]

            action, value = self.controller.feed_forward(states[idx][np.newaxis,:], goal[np.newaxis,:])
            actions.append(action[0])
            values.append(value[0])
            goals.append(goal) # Actual Goal

        return actions, values, self.goals, goals

    def train(self, exp_buffer):
        stime = time.time()
        controller_summary = self.controller.feed_backward(exp_buffer.nparray(), epochs=1)
        meta_summary = self.meta_controller.feed_backward(exp_buffer.nparray(), epochs=1)
        etime = time.time()
        print(f'\nTraining Duration: {etime-stime} sec')

        return [meta_summary, controller_summary]

    def record(self, value_dict, episode, episode_summary=None):
        summary = tf.Summary()
        for name, value in value_dict.items():
            summary.value.add(tag=name, simple_value=value)
        self.writer.add_summary(summary, episode)
        if episode_summary is not None:
            self.writer.add_summary(extra_summary, episode)
        self.writer.flush()

