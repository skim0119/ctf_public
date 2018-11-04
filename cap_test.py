import time
import gym
import gym_cap
import numpy as np

# the modules that you can use to generate the policy.
import policy.patrol 
import policy.random
import policy.roomba
import policy.policy_RL
import policy.zeros

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

done = False
t = 0
total_score = 0

# reset the environment and select the policies for each of the team
#policy_red  = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red, model_dir='model/B4R4_self_VANILLA/',color='red')
policy_red  = policy.random.PolicyGen(env.get_map, env.get_team_red)
policy_blue = policy.policy_RL.PolicyGen(env.get_map, env.get_team_blue, model_dir='model/B4R4_self_VANILLA/')
observation = env.reset(map_size=20,
                        policy_blue=policy_blue,
                        policy_red=policy_red)

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
        if t == 150:
            break

    env.reset()
    done = False
    print("Time: %.2f s, score: %.2f" %
        ((time.time() - start_time),reward))
