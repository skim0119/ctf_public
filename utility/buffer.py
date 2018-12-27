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

    def __setattr__(self, name, value):
        if self.__dict__.get(name,False):
            if name == 'depth' or name == 'length_max' or name == 'length':
                raise AttributeError('Class does not allow assigning new depth and length')
            elif name == 'buffer':
                # if buffer is assigned, change the length accordingly
                self.buffer = value
                self.length = len(self.buffer[0])
        else:
            raise AttributeError('Class does not allow creating new member')

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
        s_, e_ = self.length-serial_length, self.length
        traj_list = []
        while s_ >= 0:
            new_traj = Trajectory(depth=self.depth, length_max=self.length_max)
            new_buffer = [buf[s_:e_] for buf in self.buffer]
            new_traj.buffer = new_buffer
            traj_list.append(new_traj)

        return traj_list

class Trajectory_buffer:
    """Trajectory_buffer

    Buffer for trajectory storage and sampling
    Once the trajectory is pushed, althering trajectory would be impossible. (avoid)

    The shape of the buffer must have form [None, None, depth]
    Each depth must be unstackable, and each unstacked array will have form [None, None]+shape

    Second shape must be consist with others.

    Each trajectory is stored in list.
    At the moment of sampling, the list of individual element is returned in numpy array format.

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

    TODO:
        - Think about better method of handling 2-D trajectory elements
    """

    def __init__(self, depth=4, capacity=256):
        """__init__

        :param capacity: maximum size of the buffer.
        :param return_size: size to return
        """

        # Configuration
        self.depth = depth
        self.capacity = capacity

        # Initialize Components
        self.buffer_size = 0;
        self.buffer = [[] for _ in range(self.depth)]

    def __call__(self):
        return self.buffer

    def __repr__(self):
        return f'Trajectory Buffer(buffer capacity = {self.capacity}, return size = {self.return_size})'

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, item):
        self.buffer[index] = item

    def is_empty(self):
        return self.buffer_size == 0

    def is_full(self):
        return self.buffer_size == self.capacity

    def append(self, traj):
        for i, elem in enumerate(traj):
            self.buffer[i].append(elem)
        self.buffer_size += 1

    def extend(self, trajs):
        for traj in  trajs:
            for i, elem in enumerate(traj):
                self.buffer[i].append(elem)
        self.buffer_size += len(trajs)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            self.buffer_size = len(self.buffer)

    def sample(self, size, flush=True):
        """sample

        Return in (None,None)+shape

        :param flush: True - Emtpy the buffer after sampling
        """
        if flush:
            ret = tuple(self.buffer)
            self.buffer = [[] for _ in range(self.depth)]
        else:
            raise NotImplementedError
        return ret

if __name__ == '__main__':
    pass
