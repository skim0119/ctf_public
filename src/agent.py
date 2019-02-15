import numpy as np
import tensorflow as tf

from network import ActorCritic as Network

from utility.utils import store_args, discount_rewards
from utility.preprocessor import one_hot_encoder_v3 as state_encoder

class Agent():
    @store_args
    def __init__(self, input_shape, output_shape, lr_actor, lr_critic, name,
                 entropy_beta=0.05):
        # Define Hierarchy of Network
        # - It is double layer, but it can be multi-layer
        self.meta_controller = Network()
        self.controller = Network()
        self.network = Network(input_shape=INPUT_SHAPE,
                               output_shape=ACTION_SHAPE,
                               lr_actor=LR_A,
                               lr_critic=LR_C,
                               entropy_beta=0.05,
                               name=name,
                               new_network=new_network)
