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
        Provides method of sampling and shuffling.

Note:
    Use 'from buffer import <class_name>' to use specific method/class

Todo:
    * Finish documenting the module
    * Implement __main__ for quick debugging and checking
    * Move Experience_buffer from utils.py to this file.
        * Require to change all imports

"""


class Trajectory_buffer:
    """Trajectory_buffer

    Trajectory buffer
        Each trajectory stores tuples for MDP.
    Support returning and shuffling features
    The size of the element is adjustable. (or not hardly defined)
    Once the trajectory is pushed, althering trajectory would be impossible.

    Try to keep the storage separated for each element: avoid transposing and column-searching

    Key Terms:
        depth : number of element in each point along the trajectory.
            ex) [s0, a, r, s1] has depth 4
        shuffle : return shuffled trajectory
        keys : name of each element to keep indices. (in order)
        buffer_size : maximum size of the buffer. Infinite buffer is not allowed
        return_size : fixed size to return trajectory.

    Methods:
        __repr__
        __len__ : Return the length of the currently stored number of trajectory
        append (list)
        extend (list) : append each element in list. (serial append)
        flush : empty, reset, return remaining at once
        sample (int) : sample 'return_size' amount of trajectory

    Notes:
        - The defualt dictionary will be used to store informations.
        - Each columns is assigned with each key, and any 'return' will return in same order.
        - The shuffling uses random.shuffle() method on separate index=[0,1,2,3,4 ...] array
        - The class is originally written to use for A3C with LSTM network.
    """

    def __init__(self, keys, depth=4, buffer_max=50, shuffle=False):
        """__init__

        :param keys:
        :param depth:
        :param buffer_max:
        :param shuffle: 
        """
        # Configuration
        self.keys = keys
        self.depth = depth
        self.buffer_max = buffer_max

        # Initialize Components
        self.buffer_size = 0;
        self.buffer = defaultdict(deque)
        self.shuffle = shuffle
        if shuffle:
            self.indices = []
        
    def __repr__(self):
        return f'Trajectory Buffer(keys={self.keys},depth={self.depth},length={self.buffer_size}'

    def __len__(self):
        return self.buffer_size

    def empty(self):
        return self.buffer_size == 0

    def append(self, traj, keys=None):
        """append

        :param traj: Trajectory
        :param keys: Keys to each element in order
        """
        if keys is None:
            assert len(traj) == len(self.keys)
            keys = self.keys
        else:
            assert len(traj) == len(keys)
            assert set(keys) in (self.keys)

        for element, key in zip(traj, keys):
            self.buffer[key].append(element)
            if self.buffer_size == self.buffer_max:
                self.buffer[key].popleft()
        self.buffer_size = min(self.buffer_size+1, self.buffer_max)
    
    def extend(self, trajs, keys):
        """extend

        :param trajs: list of trajectories
        :param keys: Keys to each element in order
        """
        for traj in trajs:
            self.append(traj, keys)
    
    def flush(self, return_size=None):
        """flush
        Return the remaining buffer and reset.
        If return size is give, only return that amount of random sample

        :param return_size:
        """
        raise NotImplementedError
        batch = np.reshape(np.array(self.buffer), [len(self.buffer),self.experience_shape])
        self.buffer = []
        return batch
    
    def sample(self, return_size=4, order=False):
        """sample
        return 'return_size' number of randomly pulled trajectory
        if order is on, pull from right end first.

        :param return_size: Number of trajectory to return
        :param order:
        """
        raise NotImplementedError
        if shuffle:
            random.shuffle(self.buffer)
            
        if size > len(self.buffer):
            return np.array(self.buffer)
        else:
            #return np.array([self.buffer.pop(random.randrange(len(self.buffer))) for _ in range(size)])
            return np.reshape(np.array(random.sample(self.buffer,size)),[size,self.experience_shape])


if __name__ == '__main__':
    pass
