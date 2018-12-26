import numpy as np
import random
import collections
from collections import defaultdict, deque

"""Utility module for buffer modules in CtF Project

Please include the docstrings for any method or class to easily reference from Jupyter
Any pipeline or data manipulation is excluded: they are included in dataModule.py file.

Classes:
    :Experience_buffer: (in utils.py)

    :Trajectory Buffer: Advanced buffer used to keep the trajectory correlation between samples
        Advanced version of Experience Buffer
        The container in format of multidimension list.
        Provides method of sampling 

Note:
    Use 'from buffer import <class_name>' to use specific method/class

Todo:
    * Finish documenting the module
    * Implement __main__ for quick debugging and checking
    * Move Experience_buffer from utils.py to this file.
        * Require to change all imports

"""

class Trajectory:
    """ Trajectory

    Trajectory of [s0, a, r, s1] (or any other MDP tuples)

    Equivalent to : list [[s0 a r s1]_0, [s0 a r s1]_1, [s0 a r s1]_2, ...]
    Shape of : [None, Depth]
    Each depth must be unstackable

    Key Terms:
        depth : number of element in each point along the trajectory.
            ex) [s0, a, r, s1] has depth 4

    Methods:
        __repr__
        __len__ : Return the length of the currently stored trajectory
        is_full : boolean, whether trajectory is full
        append (list)

    Notes:
        - Trajectory is only pushed single node at a time.

    """
    def __init__(self, depth=4, length_max=150):
        # Configuration
        self.depth = depth
        self.length_max = length_max

        # Initialize Components
        self.length = 0;
        self.buffer = [[] for _ in range(depth)]

    def __call__(self):
        return self.buffer
        
    def __repr__(self):
        return f'Trajectory (depth={self.depth},length={self.length_max})'

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.buffer[index]

    def is_full(self):
        return self.length == self.length_max

    def append(self, mdp_tup):
        for i, element in enumerate(mdp_tup):
            self.buffer[i].append(element)
            if self.length == self.length_max:
                self.buffer[i] = self.buffer[i][1:]
        self.length = min(self.length+1, self.length_max)

    def trim(self, serial_length):
        if self.length < serial_length:
            return None



class Trajectory_buffer:
    """Trajectory_buffer

    Buffer for trajectory storage and sampling
    Once the trajectory is pushed, althering trajectory would be impossible. (avoid)

    The shape of the buffer must have form [None, None, depth]
    Each depth must be unstackable, and each unstacked array will have form [None, None]+shape

    Second shape must be consist with others.

    Methods:
        __repr__
        __len__ : Return the length of the currently stored number of trajectory
        is_empty : boolean, whether buffer is empty. (length == 0)
        is_full : boolean, whether buffer is full
        append (list)
        extend (list)
        flush : empty, reset, return remaining at once
        sample (int) : sample 'return_size' amount of trajectory

    Notes:
        - The sampling uses random.shuffle() method on separate index=[0,1,2,3,4 ...] array
        - The class is originally written to use for A3C with LSTM network. (Save trajectory in series)
    """

    def __init__(self, depth=4, buffer_capacity=256, return_size=8):
        """__init__

        :param buffer_capacity: maximum size of the buffer.
        :param return_size: size to return 
        """

        # Configuration
        self.depth = depth
        self.buffer_capacity = buffer_capacity
        self.return_size = return_size

        # Initialize Components
        self.buffer_size = 0;
        self.buffer = [[] for _ in range(self.depth)]
        
    def __call__(self):
        return self.buffer

    def __repr__(self):
        return f'Trajectory Buffer(buffer capacity = {self.buffer_capacity}, return size = {self.return_size})'

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, item):
        self.buffer[index] = item

    def is_empty(self):
        return self.buffer_size == 0

    def is_full(self):
        return self.buffer_size == self.buffer_capacity

    def append(self, traj):
        self.buffer.append(traj)
        self.buffer_size += 1

    def extend(self, trajs):
        self.buffer.extend(trajs)
        if len(self.buffer) > self.buffer_capacity:
            self.buffer = self.buffer[-self.buffer_capacity:]
        self.buffer_size = len(self.buffer)
    
    def sample(self, flush=True):
        """sample

        :param flush: True - Emtpy the buffer after sampling
        """
            
        if self.return_size > len(self.buffer):
            ret = self.buffer
            self.buffer = []
            return ret
        else:
            ret = random.sample(self.buffer, self.return_size)
            self.buffer = []
            return ret


        self.buffer.clear()



if __name__ == '__main__':
    pass
