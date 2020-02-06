import os
import random
import threading
import numpy as np

from keras.preprocessing.image import random_rotation

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def Generator(x, y,
              mode='train',
              input_shape=32,
              batch_size=128,
              classes=10,
              seed=42,
              shuffle=False,
              **kwargs):
    
    datalist = np.arange(x.shape[0])
    batch = 0
    X = np.zeros((batch_size, input_shape, input_shape, 3))
    Y = np.zeros((batch_size, classes))
    while True:
        if shuffle:
            random.shuffle(datalist)
            
        for data in datalist:
            img = x[data].astype('float32')
            img /= 255
            
            if mode == 'train':
                if np.random.random() > .5:
                    # vertical flip
                    img = img[::-1]
                    
                if np.random.random() > .5:
                    # horizontal flip
                    img = img[:,::-1]
                    
                img = random_rotation(img, 10, row_axis=0, col_axis=1, channel_axis=2)
                
            X[batch] = img
            Y[batch, y[data]] += 1
            
            batch += 1
            if batch >= batch_size:
                yield X, Y
                batch = 0
                X = np.zeros((batch_size, input_shape, input_shape, 3))
                Y = np.zeros((batch_size, classes))