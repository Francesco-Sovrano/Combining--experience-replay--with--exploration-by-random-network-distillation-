from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque

class TrainConfig(object):
  """Tiny config, for testing."""
  burn_in_steps = 12 
  num_lags = 2 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [2]
  num_freq = 2
  keep_prob = 1.0 # dropout
  sample_prob = 0.0 # sample ground true
  use_error_prop = True

def rnn_with_feed_prev(cell, inputs, is_training, config, initial_state=None):
    prev = None
    outputs = []
    sample_prob = config.sample_prob # scheduled sampling probability

    feed_prev = not is_training if config.use_error_prop else False
    is_sample = is_training and initial_state is not None # decoder  

    if is_sample:
        print("Creating model @ training  --> Using scheduled sampling.")
    else:
        print("Creating model @ training  --> Not using scheduled sampling.")
    
    if feed_prev:
        print(' '*30+" --> Feeding output back into input.")
    else:
        print(' '*30+" --> Feeding ground truth into input.")

    with tf.variable_scope("rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])
        burn_in_steps = config.burn_in_steps
        output_size = cell.output_size

        # phased lstm input
        inp_t = tf.expand_dims(tf.range(1,batch_size+1), 1)

        dist = Bernoulli(probs=config.sample_prob)
        samples = dist.sample(sample_shape=num_steps)
        # with tf.Session() as sess:
        #     print('bernoulli',samples.eval())
        if initial_state is None:
            initial_state = cell.zero_state(batch_size, dtype= tf.float32)
        state = initial_state

        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]
            
            if is_sample and time_step > 0: 
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool),  lambda:tf.identity(inp) , \
                       lambda:fully_connected(cell_output, input_size, activation_fn=tf.sigmoid))
                    
                    
            if feed_prev and prev is not None and time_step >= burn_in_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = fully_connected(prev, input_size,  activation_fn=tf.sigmoid)
                    #print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

            if isinstance(cell._cells[0], tf.contrib.rnn.PhasedLSTMCell):
                (cell_output, state) = cell((inp_t, inp), state)
            else:
                (cell_output, state) = cell(inp, state)

            prev = cell_output
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                outputs.append(output)

    outputs = tf.stack(outputs, 1)
    return outputs, state

def _shift (input_list, new_item):
    """Update lag number of states"""
    output_list = copy.copy(input_list)
    output_list = deque(output_list)
    output_list.append(new_item) # deque = [1, 2, 3]
    output_list.popleft() # deque =[2, 3]
    return output_list

def _list_to_states(states_list):
    """Transform a list of state tuples into an augmented tuple state
    customizable function, depends on how long history is used"""
    num_layers = len(states_list[0])# state = (layer1, layer2...), layer1 = (c,h), c = tensor(batch_size, num_steps)
    output_states = ()
    for layer in range(num_layers):
        output_state = ()
        for states in states_list:
                #c,h = states[layer] for LSTM
                output_state += (states[layer],)
        output_states += (output_state,)
        # new cell has s*num_lags states
    return output_states

def tensor_rnn_with_feed_prev(cell, inputs, is_training, config, initial_states=None):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    prev = None
    feed_prev = not is_training if config.use_error_prop else False
    is_sample = is_training and initial_states is not None

    if is_sample:
        print("Creating model @ training  --> Using scheduled sampling.")
    else:
        print("Creating model @ training  --> Not using scheduled sampling.")
    
    if feed_prev:
        print(' '*30+" --> Feeding output back into input.")
    else:
        print(' '*30+" --> Feeding ground truth into input.")

    with tf.variable_scope("trnn") as varscope:
        if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])
        output_size = cell.output_size
        burn_in_steps =  config.burn_in_steps
        
        # Scheduled sampling
        dist = Bernoulli(probs=config.sample_prob)
        samples = dist.sample(sample_shape=num_steps)
        
        if initial_states is None:
            initial_states =[]
            for lag in range(config.num_lags):
                initial_state =  cell.zero_state(batch_size, dtype= tf.float32)
                initial_states.append(initial_state)

        states_list = initial_states #list of high order states
    
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]

            if is_sample and time_step > 0: 
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool),  lambda:tf.identity(inp) , \
                       lambda:fully_connected(cell_output, input_size, activation_fn=tf.sigmoid))
                    
            if feed_prev and prev is not None and time_step >= burn_in_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                    #print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

            states = _list_to_states(states_list)
            """input tensor is [batch_size, num_steps, input_size]"""
            (cell_output, state)=cell(inp, states)

            # dropout 
            # keep_prob = tf.placeholder(tf.float32)
            keep_prob = 0.5
            cell_output = tf.nn.dropout(cell_output, keep_prob)

            states_list = _shift(states_list, state)

            prev = cell_output
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                outputs.append(output)

    outputs = tf.stack(outputs,1)
    return outputs, states_list



