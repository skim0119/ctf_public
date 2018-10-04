import numpy as np
import gym_cap.envs.const as CONST

UNKNOWN  = CONST.UNKNOWN # -1
TEAM1_BG = CONST.TEAM1_BACKGROUND # 0
TEAM2_BG = CONST.TEAM2_BACKGROUND # 1
TEAM1_AG = CONST.TEAM1_UGV # 2
TEAM2_AG = CONST.TEAM2_UGV # 4
TEAM1_FL = CONST.TEAM1_FLAG # 6
TEAM2_FL = CONST.TEAM2_FLAG # 7
OBSTACLE = CONST.OBSTACLE # 8
DEAD     = CONST.DEAD # 9
SELECTED = CONST.SELECTED # 10
COMPLETED= CONST.COMPLETED # 11

VISION_RANGE = 10 #CONST.UGV_RANGE # Regardless, the vision will be 20x20
VISION_dX    = 2*VISION_RANGE+1
VISION_dY    = 2*VISION_RANGE+1

def one_hot_encoder(state, agents):
    ret = np.zeros((len(agents),VISION_dX,VISION_dY,6))
    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = {UNKNOWN:0, DEAD:0,
                   TEAM1_BG:1, TEAM2_BG:1,
                   TEAM1_AG:2, TEAM2_AG:2,
                   3:3, 5:3, # UAV, does not need to be included for now
                   TEAM1_FL:4, TEAM2_FL:4,
                   OBSTACLE:5}
    map_color   = {UNKNOWN:1, DEAD:0, OBSTACLE:1,
                   TEAM1_BG:1, TEAM2_BG:-1,
                   TEAM1_AG:1, TEAM2_AG:-1,
                   3:1, 5:-1, # UAV, does not need to be included for now
                   TEAM1_FL:1, TEAM2_FL:-1}
    
    # Expand the observation with 3-thickness wall
    # - in order to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.ones((sx+2*VISION_RANGE, sy+2*VISION_RANGE)) * OBSTACLE # 8 for obstacle
    _state[VISION_RANGE:VISION_RANGE+sx, VISION_RANGE:VISION_RANGE+sy] = state
    state = _state

    for idx,agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += VISION_RANGE
        y += VISION_RANGE
        vision = state[x-VISION_RANGE:x+VISION_RANGE+1,y-VISION_RANGE:y+VISION_RANGE+1] # extract the limited view for the agent (5x5)
        for i in range(len(vision)):
            for j in range(len(vision[0])):
                if vision[i][j] != -1:
                    channel = map_channel[vision[i][j]]
                    ret[idx][i][j][channel] = map_color[vision[i][j]]
    return ret