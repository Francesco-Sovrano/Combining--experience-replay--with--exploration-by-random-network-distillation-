# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
from tensorflow.contrib.rnn import LSTMBlockCell, LSTMBlockFusedCell, GRUBlockCellV2
from utils.tensorflow_utils import gpu_count

class RNN(object):
	
	def __init__(self, type, units, batch_size, direction=1, dtype='float32', stack_size=1, dropout=0., training=True):
		self.training = training
		self.stack_size = stack_size
		self.tf_dtype = eval('tf.{}'.format(dtype))
		self.np_dtype = eval('np.{}'.format(dtype))
		self.batch_size = batch_size
		self.units = units
		self.type = type
		self.use_gpu = gpu_count() > 0
		# When you calculate the accuracy and validation, you need to manually set the keep_probability to 1 (or the dropout to 0) so that you don't actually drop any of your weight values when you are evaluating your network. If you don't do this, you'll essentially miscalculate the value you've trained your network to predict thus far. This could certainly negatively affect your acc/val scores. Especially with a 50% dropout_probability rate.
		self.dropout_probability = dropout if not self.training else 0.
		self.direction = direction
		self.rnn_layer = None
		if type == 'LSTM':
			self.cell = CudnnLSTM if self.use_gpu else LSTMBlockCell
		elif type == 'GRU':
			self.cell = CudnnGRU if self.use_gpu else GRUBlockCellV2
		# State shape
		if self.use_gpu:
			# initial_state: a tuple of tensor(s) of shape [num_layers * num_dirs, batch_size, num_units]
			if self.type == 'LSTM':
				self.state_shape = [2, self.stack_size*self.direction, self.batch_size, self.units]
			elif type == 'GRU':
				self.state_shape = [1, self.stack_size*self.direction, self.batch_size, self.units]
		else:
			if self.type == 'LSTM':
				self.state_shape = [self.direction, self.stack_size, 2, self.batch_size, self.units]
			elif type == 'GRU':
				self.state_shape = [self.direction, self.stack_size, self.batch_size, self.units]
		
	def default_state(self):
		return np.zeros(self.state_shape, self.np_dtype)
		
	def state_placeholder(self, name=''):
		return tf.placeholder(shape=[None]+self.state_shape, name='{}_{}'.format(name,self.type), dtype=self.tf_dtype)
	
	def _process_single_batch(self, input, initial_state):
		# Build initial state
		if self.use_gpu:
			# initial_state: a tuple of tensor(s) of shape [num_layers * num_dirs, batch_size, num_units]
			initial_state = tuple(tf.unstack(initial_state))
		else:
			state_list = []
			if self.type == 'LSTM':
				for d in range(self.direction):
					state = initial_state[d]
					state_list.append([
						tf.nn.rnn_cell.LSTMStateTuple(state[i][0],state[i][1]) 
						for i in range(self.stack_size)
					])
			else:
				for d in range(self.direction):
					state_list.append(tf.unstack(initial_state[d]))
			initial_state = state_list
		output, final_state = self.rnn_layer(inputs=input, initial_state=initial_state)
		return output, final_state
	
	def process_batches(self, input, initial_states, sizes):
		# Add batch dimension, sequence length here is input batch size, while sequence batch size is batch_size
		if len(input.get_shape()) > 2:
			input = tf.layers.flatten(input)
		input_depth = input.get_shape().as_list()[-1]
		input = tf.reshape(input, [-1, self.batch_size, int(input_depth/self.batch_size)])
		# Build RNN layer
		if self.rnn_layer is None:
			self.rnn_layer = self._build(input)
		# Get loop constants
		batch_limits = tf.concat([[0],tf.cumsum(sizes)], 0)
		batch_count = tf.shape(sizes)[0]
		output_shape = [self.batch_size, self.direction*self.units]
		# Build condition
		condition = lambda i, output, final_states: i < batch_count
		# Build body
		def body(i, output, final_states):
			start = batch_limits[i]
			end = batch_limits[i+1]
			ith_output, ith_final_state = self._process_single_batch(input=input[start:end], initial_state=initial_states[i])
			output = tf.concat((output,ith_output), 0)
			final_states = tf.concat((final_states,[ith_final_state]), 0)
			return i+1, output, final_states
		# Build input
		i = 0
		output = tf.zeros([1]+output_shape, self.tf_dtype)
		final_states = tf.zeros([1]+self.state_shape, self.tf_dtype)
		# Loop
		_, output, final_states = tf.while_loop(
			cond=condition, # A callable that represents the termination condition of the loop.
			body=body, # A callable that represents the loop body.
			loop_vars=[
				i,  # 1st variable # batch index
				output, # 2nd variable # RNN outputs
				final_states # 3rd variable # RNN final states
			],
			shape_invariants=[ # The shape invariants for the loop variables.
				tf.TensorShape(()), # 1st variable
				tf.TensorShape([None]+output_shape), # 2nd variable
				tf.TensorShape([None]+self.state_shape) # 3rd variable
			],
			swap_memory=True, # Whether GPU-CPU memory swap is enabled for this loop
			return_same_structure=True # If True, output has same structure as loop_vars.
		)
		output = tf.layers.flatten(output)
		# Return result
		return output[1:], final_states[1:] # Remove first element we used to allow concatenations inside loop body	
		
	def _build(self, inputs):
		if self.use_gpu:
			rnn = self.cell(
				num_layers=self.stack_size, 
				num_units=self.units, 
				direction='unidirectional' if self.direction == 1 else 'bidirectional',
				dropout=self.dropout_probability, # set to 0. for no dropout 
				dtype=self.tf_dtype
			)
			rnn.build(inputs.get_shape()) # Build now
			def rnn_layer(inputs, initial_state):
				# Do not build here, just call
				return rnn.call(inputs=inputs, initial_state=initial_state, training=self.training)
			return rnn_layer
		else:
			# Build RNN cells
			cell_list = []
			for d in range(self.direction):
				cell_list.append([self.cell(num_units=self.units, name='{}_cell{}_{}'.format(self.type,d,i)) for i in range(self.stack_size)])
			# Apply dropout_probability
			if self.dropout_probability > 0.:
				dropout = tf.nn.rnn_cell.DropoutWrapper
				for rnn_cells in cell_list:
					rnn_cells = [
						dropout(cell=cell, output_keep_prob=1-self.dropout_probability) 
						for cell in rnn_cells[:-1]
					] + rnn_cells[-1:] # Do not apply dropout_probability to last layer, as in GPU counterpart implementation
			# Build stacked dynamic RNN <https://stackoverflow.com/questions/49242266/difference-between-multirnncell-and-stack-bidirectional-dynamic-rnn-in-tensorflo>
			# sequence_length = [tf.shape(inputs)[0]]
			if self.direction == 1: # Unidirectional
				def unidirectional_rnn_layer(inputs, initial_state):
					output = inputs
					final_state = []
					for i,(cell,state) in enumerate(zip(cell_list[0],initial_state[0])):
						output, output_state = tf.nn.dynamic_rnn(
							cell = cell,
							inputs = output,
							initial_state = state,
							# sequence_length = sequence_length,
							time_major = True # The shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size, depth]. If false, these Tensors must be shaped [batch_size, max_time, depth]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.
						)
						final_state.append(output_state)
					final_state = tf.reshape([final_state], self.state_shape)
					return output, final_state
				return unidirectional_rnn_layer
			else: # Bidirectional
				def bidirectional_rnn_layer(inputs, initial_state):
					output, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
						cells_fw = cell_list[0], # List of instances of RNNCell, one per layer, to be used for forward direction.
						cells_bw = cell_list[1], # List of instances of RNNCell, one per layer, to be used for backward direction.
						inputs = inputs, # The RNN inputs. this must be a tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.
						initial_states_fw = initial_state[0], # (optional) A list of the initial states (one per layer) for the forward RNN. Each tensor must has an appropriate type and shape [batch_size, cell_fw.state_size].
						initial_states_bw = initial_state[1], # (optional) Same as for initial_states_fw, but using the corresponding properties of cells_bw.
						# sequence_length = sequence_length, # (optional) An int32/int64 vector, size [batch_size], containing the actual lengths for each of the sequences.
						time_major = True # The shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size, depth]. If false, these Tensors must be shaped [batch_size, max_time, depth]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.
					)
					final_state = [output_state_fw, output_state_bw]
					final_state = tf.reshape(final_state, self.state_shape)
					return output, final_state
				return bidirectional_rnn_layer
