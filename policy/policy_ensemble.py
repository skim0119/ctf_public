import numpy as np
import tensorflow as tf

from utility.utils import store_args
from utility.dataModule import one_hot_encoder, one_hot_encoder_v2
from utility.RL_Wrapper import TrainedNetwork


class PolicyGen:
    @store_args
    def __init__(
        self,
        env_map=None,
        team=None,
        color='blue',
        vision_range=9,
        fix_policy=[2, 2],
        *args,
        **kwargs
    ):
        self.is_blue = (color == 'blue')
        self.policy = [
            TrainedNetwork(model_name='sub_A3C_nav'),
            TrainedNetwork(model_name='A3C_att')
        ]

    def gen_action(self, agent_list, observation, free_map=None):
        state = one_hot_encoder(observation, agent_list, self.vision_range, reverse=not self.is_blue)
        state_wide = one_hot_encoder_v2(observation, agent_list, 19, reverse=not self.is_blue)
        p = self.policy
        choices = [p[0].get_action(state), p[1].get_action(state_wide)]
        # choices = [p.get_action(state) for p in self.policy]

        # Arbitrary
        action_out = []
        si, ei = 0, 0
        for pid, n in enumerate(self.fix_policy):
            ei += n
            action_out.extend(choices[pid][si:ei])
            si = ei

        return action_out
