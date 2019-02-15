import numpy as np
import random
import scipy.signal

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


def discount_rewards(rewards, gamma, normalize=False, mask_array=None):
    """ take 1D float numpy array of rewards and compute discounted reward

    Args:
        rewards (numpy.array): list of rewards.
        gamma (float): discount rate
        normalize (bool): If true, normalize at the end (default=False)

    Returns:
        numpy.list : Return discounted reward

    """

    if mask_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]
    else:
        y, adv = 0.0, []
        mask_reverse = mask_array[::-1]
        for i, reward in enumerate(reversed(rewards)):
            y = reward + gamma * y * (1 - mask_reverse[i])
            adv.append(y)
        disc_r = np.array(adv)[::-1]

        if normalize:
            disc_r = (disc_r - np.mean(disc_r)) / (np.std(disc_r) + 1e-13)

        return disc_r


def normalize(r):
    """ take 1D float numpy array and normalize it

    Args:
        r (numpy.array): list of numbers

    Returns:
        numpy.list : return normalized list

    """

    return (r - np.mean(r)) / (np.std(r) + 1e-13)  # small addition to avoid dividing by zero


def retrace(targets, behaviors, lambda_=0.2):
    """ take target and behavior policy values, and return the retrace weight

    Args:
        target (1d array float): list of target policy values in series
            (policy of target network)
        behavior (1d array float): list of target policy values in sequence
            (policy of behavior network)
        lambda_ (float): retrace coefficient

    Returns:
        weight (1D list)          : return retrace weight
    """

    weight = []
    ratio = targets / behaviors
    for r in ratio:
        weight.append(lambda_ * min(1.0, r))

    return np.array(weight)


def retrace_prod(targets, behaviors, lambda_=0.2):
    """ take target and behavior policy values, and return the cumulative product of weights

    Args:
        target (1d array float): list of target policy values in series
            (policy of target network)
        behavior (1d array float): list of target policy values in sequence
            (policy of behavior network)
        lambda_ (float): retrace coefficient

    Returns:
        weight_cumulate (1D list) : return retrace weight in cumulative-product
    """

    return np.cumprod(retrace(targets, behaviors, lambda_))


class MovingAverage:
    """MovingAverage
    Container that only store give size of element, and store moving average.
    Queue structure of container.
    """

    def __init__(self, size):
        """__init__

        :param size: number of element that will be stored in he container
        """
        from collections import deque
        self.average = 0.0
        self.size = size

        self.queue = deque(maxlen=size)

    def __call__(self):
        """__call__"""
        return self.average

    def tolist(self):
        """tolist
        Return the elements in the container in (list) structure
        """
        return list(self.queue)

    def extend(self, l: list):
        """extend

        Similar to list.extend

        :param l (list): list of number that will be extended in the deque
        """
        # Append list of numbers
        self.queue.extend(l)
        self.size = len(self.queue)
        self.average = sum(self.queue) / self.size

    def append(self, n):
        """append

        Element-wise appending in the container

        :param n: number that will be appended on the container.
        """
        s = len(self.queue)
        if s == self.size:
            self.average = ((self.average * self.size) - self.queue[0] + n) / self.size
        else:
            self.average = (self.average * s + n) / (s + 1)
        self.queue.append(n)

    def clear(self):
        """clear
        reset the container
        """
        self.average = 0.0
        self.queue.clear()


class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s_, g, t, grip_info\')'
        assert type(experience[5]) == bool

        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer.
        # Only removing a single transition slows down performance
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size / 6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=batch_size)

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals

def store_args(method):
    """Stores provided method args as instance attributes.
    Sourced: Baselines
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

if __name__ == '__main__':
    pass
