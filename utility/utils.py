from collections import deque 
import numpy as np
import random

def discount_rewards(r, gamma, normalize=False):
    """ take 1D float numpy array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(r.size)):
        running_add = (running_add * gamma + r[t])
        discounted_r[t] = running_add
    if normalize:
        discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r)+1e-8) # normalize
    return discounted_r

def normalize(r):
    return (r - np.mean(r)) / (np.std(r)+1e-8)

class MovingAverage:
    def __init__(self, size):
        self.ma = 0.0
        self.size = size
        
        self.queue = deque(maxlen=size)
        
    def __call__(self):
        return self.ma

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
