from collections import deque 
import numpy as np
import random

"""Utility methods and classes used in CtF Problem

This module contains extra features and functions frequently used in Ctf Project.
Please include the docstrings for any method or class to easily reference from Jupyter
Any pipeline or data manipulation is excluded: they are included in dataModule.py file.

Methods:
    discount_rewards(numpy.list, float, bool): 
        Perform discounting reward to the list by the ratio 'gamma'
        Include normalization of the list at the end.

    normalize(numpy.list):
        Only perform the normalization of the list.
        * Centralize by subtracting the average, zoom by dividing the variance.

Classes:
    MovingAverage:
        The container in format of queue.
        Only store upto fixed amount of data, and return the average.
        Any abundant records will be removed, and the average will be kept

    Experience_bufffer:
        The container in format of list.
        Provides method of sampling and shuffling.

Note:
    Use 'from utils import <def_name>' to use specific method/class

Todo:
    * Finish documenting the module
    * If necessary, include __main__ in module for debuging and quick operation checking

"""

def discount_rewards(rewards, gamma, normalize=False):
    """ take 1D float numpy array of rewards and compute discounted reward 

    Args:
        rewards (numpy.array): list of rewards.
        gamma (float): discount rate
        normalize (bool): If true, normalize at the end (default=False) 

    Returns:
        numpy.list : Return discounted reward
    
    """

    discReward = np.zeros_like(r)
    runningAdd = 0.0
    for idx, reward in enumerate(reversed(rewards)):
        runningAdd = (runningAdd * gamma + reward)
        discReward[idx] = runningAdd

    if normalize:
        discReward = (discReward - np.mean(discReward)) / (np.std(discReward)+1e-8) # normalize

    return discReward

def normalize(r):
    """ take 1D float numpy array and normalize it

    Args:
        r (numpy.array): list of numbers

    Returns:
        numpy.list : return normalized list
    
    """

    return (r - np.mean(r)) / (np.std(r)+1e-8) # small addition to avoid dividing by zero

class MovingAverage:
    def __init__(self, size):
        self.ma = 0.0
        self.size = size
        
        self.queue = deque(maxlen=size)
        
    def __call__(self):
        return self.ma
    
    def tolist(self):
        return list(self.queue)

    def extend(self, l:list):
        # Append list of numbers
        self.queue.extend(l)
        self.size = len(self.queue)
        self.ma = sum(self.queue) / self.size
        
    def append(self, n):
        s = len(self.queue)
        if s == self.size:
            self.ma = ((self.ma * self.size) - self.queue[0] + n) / self.size
        else:
            self.ma = (self.ma * s + n) / (s + 1)
        self.queue.append(n)

class Experience_buffer():
    def __init__(self, experience_shape=4, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.experience_shape = experience_shape
        
    def __len__(self):
        return len(self.buffer)
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
    
    def add_element(self, sample):
        self.buffer.append(sample)
    
    def flush(self):
        # Return the remaining buffer and reset.
        batch = np.reshape(np.array(self.buffer), [len(self.buffer),self.experience_shape])
        self.buffer = []
        return batch
    
    def empty(self):
        return len(self.buffer)==0
    
    def sample(self, size=2000, shuffle=False):
        if shuffle:
            random.shuffle(self.buffer)
            
        if size > len(self.buffer):
            return np.array(self.buffer)
        else:
            #return np.array([self.buffer.pop(random.randrange(len(self.buffer))) for _ in range(size)])
            return np.reshape(np.array(random.sample(self.buffer,size)),[size,self.experience_shape])
        
    def pop(self, size, shuffle=False):
        # Pop the first `size` items in order (queue).
        if shuffle:
            random.shuffle(self.buffer)
            
        i = min(len(self.buffer), size)
        batch = np.reshape(np.array(self.buffer[:i]), [i,self.experience_shape])
        self.buffer = self.buffer[i:]
        return batch 
        
if __name__ == '__main__':
    pass
