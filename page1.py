# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_lenght = 50000
truncated_backprop_lenght = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_lenght//batch_size//truncated_backprop_lenght

def generateData():
    x = np.array(np.random.choice(2, total_series_lenght, p=[0,5][0,5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return  (x, y)  

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_lenght])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_lenght])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

w = tf.Variables(np.random.rand(state_size+1, state_size), dtype = tf.float32)
b = tf.Variables(np.zeros(1, num_classes), dtype = tf.float32)

w2 = tf.Variables(np.random.rand(state_size+1, state_size), dtype = tf.float32)
b2 = tf.Variables(np.zeros(1, num_classes), dtype = tf.float32)

#unpack columns

input_series = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder, axis = 1)

#forward pass

current_state = init_state
state_series = []
for current_state input_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat(1, [current_input, current_state]) #incrising number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, w)+b) #broadcast adition
    state_series.append(next_state)
    current_state = next_state


