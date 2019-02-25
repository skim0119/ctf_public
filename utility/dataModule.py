import numpy as np
import gym_cap.envs.const as CONST

UNKNOWN = CONST.UNKNOWN            # -1
TEAM1_BG = CONST.TEAM1_BACKGROUND  # 0
TEAM2_BG = CONST.TEAM2_BACKGROUND  # 1
TEAM1_AG = CONST.TEAM1_UGV         # 2
TEAM1_UAV = CONST.TEAM1_UAV        # 3
TEAM2_AG = CONST.TEAM2_UGV         # 4
TEAM2_UAV = CONST.TEAM2_UAV        # 5
TEAM1_FL = CONST.TEAM1_FLAG        # 6
TEAM2_FL = CONST.TEAM2_FLAG        # 7
OBSTACLE = CONST.OBSTACLE          # 8
DEAD = CONST.DEAD                  # 9
SELECTED = CONST.SELECTED          # 10
COMPLETED = CONST.COMPLETED        # 11


class fake_agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_loc(self):
        return (self.x, self.y)


def state_processor(state, agents=None, vision_radius=19, full_state=None, flatten=True, reverse=False, partial=True):
    """ pre_processor

    Return encoded state, position, and goal position
    """
    if not partial:
        state = full_state
    # Find Flag Location
    flag_id = TEAM1_FL if reverse else TEAM2_FL
    flag_locs = list(zip(*np.where(full_state == flag_id)))  
    if len(flag_locs) == 0:
        flag_loc = (-1,-1)
    else:
        flag_loc = flag_locs[0]

    # One-hot encode state
    oh_state = one_hot_encoder(state, agents, vision_radius, reverse, flatten=flatten)

    # gps state
    agents_loc = [agent.get_loc() for agent in agents]

    return oh_state, agents_loc, [flag_loc]*len(agents)


def one_hot_encoder(state, agents=None, vision_radius=9, reverse=False,
                    flatten=False, locs=None):
    """Encoding pipeline for CtF state to one-hot representation

    6-channel one-hot representation of state.
    State is not binary: team2 is represented with -1.
    Channels are not symmetrical.

    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)`
    :param reverse: Reverse the color. Used for red-perspective (default=False)
    :param flatten: Return flattened representation (for array output)
    :param locs: Provide locations instead of agents. (agents must be None)

    :return oh_state: One-hot encoded state
    """
    if agents is None:
        assert locs is not None
        agents = [fake_agent(x, y) for x, y in locs]

    vision_lx = 2 * vision_radius + 1
    vision_ly = 2 * vision_radius + 1
    oh_state = np.zeros((len(agents), vision_lx, vision_ly, 6), np.float)

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = {UNKNOWN: 0, DEAD: 0,
                   TEAM1_BG: 1, TEAM2_BG: 1,
                   TEAM1_AG: 2, TEAM2_AG: 2,
                   TEAM1_UAV: 3, TEAM2_UAV: 3,
                   TEAM1_FL: 4, TEAM2_FL: 4,
                   OBSTACLE: 5}
    if not reverse:
        map_color = {UNKNOWN: 1, DEAD: 0,
                     TEAM1_BG: 0, TEAM2_BG: 1,
                     TEAM1_AG: 1, TEAM2_AG: -1,
                     TEAM1_UAV: 1, TEAM2_UAV: -1,
                     TEAM1_FL: 1, TEAM2_FL: -1,
                     OBSTACLE: 1}
    else:  # reverse color
        map_color = {UNKNOWN: 1, DEAD: 0,
                     TEAM1_BG: 1, TEAM2_BG: 0,
                     TEAM1_AG: -1, TEAM2_AG: 1,
                     TEAM1_UAV: -1, TEAM2_UAV: 1,
                     TEAM1_FL: -1, TEAM2_FL: 1,
                     OBSTACLE: 1}

    # Expand the observation with wall to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.full((sx + 2 * vision_radius, sy + 2 * vision_radius), OBSTACLE)
    _state[vision_radius:vision_radius + sx, vision_radius:vision_radius + sy] = state
    state = _state

    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x - vision_radius:x + vision_radius + 1, y - vision_radius:y + vision_radius + 1]  # extract view

        # FULL MATRIX OPERATION
        for channel, val in map_color.items():
            if val == 1:
                oh_state[idx, :, :, map_channel[channel]] += (vision == channel).astype(np.int32)
            elif val == -1:
                oh_state[idx, :, :, map_channel[channel]] -= (vision == channel).astype(np.int32)

    if flatten:
        return np.reshape(oh_state, (len(agents), -1))
    else:
        return oh_state


def one_hot_encoder_v2(state, agents, vision_radius=9, reverse=False, flatten=False):
    """ Encoding pipeline for CtF state to one-hot representation
    11-channel one-hot representation of state.
    State is binary.
    Some optimization is included.

    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)`
    :param reverse:Reverse the color. Used for red-perspective (default=False)

    :return oh_state: One-hot encoded state
    """

    num_channel = 11

    # Map channel for each elements
    if not reverse:
        order = [UNKNOWN, OBSTACLE, TEAM1_BG, TEAM2_BG, TEAM1_AG, TEAM2_AG,
                 TEAM1_UAV, TEAM2_UAV, TEAM1_FL, TEAM2_FL, DEAD]
    else:
        order = [UNKNOWN, OBSTACLE, TEAM2_BG, TEAM1_BG, TEAM2_AG, TEAM1_AG,
                 TEAM2_UAV, TEAM1_UAV, TEAM2_FL, TEAM1_FL, DEAD]
    map_channel = dict(zip(order, range(num_channel)))

    # Padding Boundary
    sx, sy = state.shape
    _state = np.full((sx + 2 * vision_radius, sy + 2 * vision_radius), OBSTACLE)
    _state[vision_radius:vision_radius + sx, vision_radius:vision_radius + sy] = state
    state = _state

    each_agent = []
    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x - vision_radius:x + vision_radius + 1, y - vision_radius:y + vision_radius + 1]  # extract view

        # operation
        each_channel = []
        for element, channel in map_channel.items():
            each_channel.append(vision == element)
        each_agent.append(np.stack(each_channel, axis=-1))
    oh_state = np.stack(each_agent, axis=0)

    if flatten:
        return np.reshape(oh_state, (len(agents), -1))
    else:
        return oh_state


def debug():
    """debug
    Include testing code for above methods and classes.
    The execution will start witn __main__, and call this method.
    """

    import gym
    import time
    env = gym.make("cap-v0")
    s = env.reset(map_size=20)

    print('start running')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')

    s = env.reset(map_size=20)

    print('start running v2')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder_v2(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')


if __name__ == '__main__':
    debug()
