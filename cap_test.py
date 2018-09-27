import time
import gym
import gym_cap
import numpy as np

# the modules that you can use to generate the policy.
import policy.patrol 
import policy.random
import policy.simple # custon written policy
import policy.policy_RL

import tensorflow as tf

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

done = False
t = 0
total_score = 0

with tf.Session() as sess:
    # reset the environment and select the policies for each of the team
    #policy_blue = policy.random.PolicyGen(env.get_map, env.get_team_blue)
    policy_red  = policy.random.PolicyGen(env.get_map, env.get_team_red)
    policy_blue = policy.policy_RL.PolicyGen(env.get_map, env.get_team_blue,sess)
    observation = env.reset(map_size=20,
                            render_mode="env",
                            policy_blue=policy_blue,
                            policy_red=policy_red)
                            #policy_red=policy.simple.PolicyGen(env.get_map, env.get_team_red))

    pre_score = 0;
    while True:
        t=0
        while not done:
            
            #you are free to select a random action
            # or generate an action using the policy
            # or select an action manually
            # and the apply the selected action to blue team
            # or use the policy selected and provided in env.reset 
            #action = env.action_space.sample()  # choose random action
            #action = [0, 0, 0, 0]
            action = policy_blue.gen_action(env.get_team_blue, env._env)
            observation, reward, done, info = env.step(action)
            
            #observation, reward, done, info = env.step()  # feedback from environment
            #print(reward-pre_score, ' ',done)
            #pre_score = reward;
            
            # render and sleep are not needed for score analysis
            t += 1
            env.render(mode="fast")
            time.sleep(.04)
            
            if t >= 100:
                break
            
        #total_score += reward
        env.reset()
        done = False

        print("Total time: %s s, score: %s" % ((time.time() - start_time),total_score))