# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from agent.network.network import Network
from utils.distributions import Categorical, Normal
from utils.rnn import RNN
import options
flags = options.get()

def is_continuous_control(policy_depth):
	return policy_depth <= 1

class Base_Network(Network):
	def __init__(self, id, qvalue_estimation, policy_heads, batch_dict, scope_dict, training=True, value_count=1, state_scaler=1):
		super().__init__(id, training)
		self.value_count = value_count
		# scope names
		self.scope_name = scope_dict['self']
		self.parent_scope_name = scope_dict['parent']
		self.sibling_scope_name = scope_dict['sibling']
		# Shape network
		states = batch_dict['state']
		self.state_batch = [s for s in states if len(s.get_shape())==4]
		self.concat_batch = [s for s in states if len(s.get_shape())!=4]
		self.size_batch = batch_dict['size']
		if flags.intrinsic_reward:
			self.training_state = batch_dict['training_state']
		self.parameters_type = eval('tf.{}'.format(flags.parameters_type))
		self.policy_heads = policy_heads 
		self.qvalue_estimation = qvalue_estimation
		self.state_scaler = state_scaler
		
	def build(self, has_actor=True, has_critic=True, use_internal_state=True, name='default'):
		print( "	[{}]Building partition {} with has_actor={}, has_critic={}, use_internal_state={}".format(self.id, name, has_actor, has_critic, use_internal_state) )
		print( "	[{}]Parameters type: {}".format(self.id, flags.parameters_type) )
		print( "	[{}]Algorithm: {}".format(self.id, flags.algorithm) )
		print( "	[{}]Network configuration: {}".format(self.id, flags.network_configuration) )
		# [CNN]
		input = [
			self._cnn_layer(name=i, input=substate_batch/self.state_scaler, scope=self.parent_scope_name)
			for i,substate_batch in enumerate(self.state_batch)
		]
		input = [
			tf.layers.flatten(i)
			for i in input
		]
		input = tf.concat(input, -1)
		print( "	[{}]CNN layer output shape: {}".format(self.id, input.get_shape()) )
		# [Training state]
		if flags.intrinsic_reward:
			input = self._weights_layer(input=input, weights=self.training_state, scope=self.scope_name)
			print( "	[{}]Weights layer output shape: {}".format(self.id, input.get_shape()) )
		# [Concat]
		if len(self.concat_batch) > 0:
			self.concat_batch = tf.concat(self.concat_batch, -1)
			input = self._concat_layer(input=input, concat=self.concat_batch, scope=self.scope_name)
			print( "	[{}]Concat layer output shape: {}".format(self.id, input.get_shape()) )
		# [RNN]
		if use_internal_state:
			self.use_internal_state = True
			input, internal_state_tuple = self._rnn_layer(input=input, scope=self.scope_name)
			self.internal_initial_state, self.internal_default_state, self.internal_final_state = internal_state_tuple 
			print( "	[{}]RNN layer output shape: {}".format(self.id, input.get_shape()) )
		# [Policy]
		self.policy_batch = self._policy_layer(input=input, scope=self.scope_name) if has_actor else None
		# print( "	[{}]Policy shape: {}".format(self.id, self.policy_batch.get_shape()) )
		# [Value]
		self.value_batch = self._value_layer(input=input, scope=self.scope_name) if has_critic else None
		# print( "	[{}]Value shape: {}".format(self.id, self.value_batch.get_shape()) )
		return self.policy_batch, self.value_batch
	
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.conv2d(name='CNN_Conv1', inputs=input, filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.layers.conv2d(name='CNN_Conv2', inputs=input, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.layers.conv2d(name='CNN_Conv3', inputs=input, filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _weights_layer(self, input, weights, scope, name="", share_trainables=True):
		layer_type = 'Weights'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			kernel = tf.stop_gradient(weights['kernel'])
			# kernel = tf.transpose(kernel, [1, 0])
			kernel = tf.layers.dense(name='Concat_Dense0', inputs=kernel, units=1, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			kernel = tf.reshape(kernel, [-1])
			bias = tf.stop_gradient(weights['bias'])
			bias = tf.reshape(bias, [-1])
			weight_state = tf.concat((kernel, bias), -1)
			input = tf.layers.flatten(input)
			input = tf.map_fn(fn=lambda b: tf.concat((b,weight_state),-1), elems=input)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
	
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=64, activation=tf.nn.elu, kernel_initializer=tf.initializers.variance_scaling)
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
		
	def _value_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'Value'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			if self.qvalue_estimation:
				policy_depth = sum(h['depth'] for h in self.policy_heads)
				policy_size = sum(h['size'] for h in self.policy_heads)
				units = policy_size*max(1,policy_depth)
				output = [
					tf.layers.dense(name='Value_Q{}_Dense1'.format(i), inputs=input, units=units, activation=None, kernel_initializer=tf.initializers.variance_scaling)
					for i in range(self.value_count)
				]
				output = tf.stack(output)
				output = tf.transpose(output, [1, 0, 2])
				if policy_size > 1 and policy_depth > 1:
					output = tf.reshape(output, [-1,self.value_count,policy_size,policy_depth])
			else:
				output = [ # Keep value heads separated
					tf.layers.dense(name='Value_V{}_Dense1'.format(i), inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling)
					for i in range(self.value_count)
				]
				output = tf.stack(output)
				output = tf.transpose(output, [1, 0, 2])
				output = tf.layers.flatten(output)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return output
			
	def _policy_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'Policy'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			output_list = []
			for h,policy_head in enumerate(self.policy_heads):
				policy_depth = policy_head['depth']
				policy_size = policy_head['size']
				if is_continuous_control(policy_depth):
					# build mean
					mu = tf.layers.dense(name='Policy_Mu_Dense{}'.format(h), inputs=input, units=policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling) # in (-inf,inf)
					# build standard deviation
					sigma = tf.layers.dense(name='Policy_Sigma_Dense{}'.format(h), inputs=input, units=policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling) # in (-inf,inf)
					# clip mu and sigma to avoid numerical instabilities
					clipped_mu = tf.clip_by_value(mu, -1,1) # in [-1,1]
					clipped_sigma = tf.clip_by_value(tf.abs(sigma), 1e-4,1) # in [1e-4,1] # sigma must be greater than 0
					# build policy batch
					policy_batch = tf.stack([clipped_mu, clipped_sigma])
					policy_batch = tf.transpose(policy_batch, [1, 0, 2])
				else: # discrete control
					policy_batch = tf.layers.dense(name='Policy_Logits_Dense{}'.format(h), inputs=input, units=policy_size*policy_depth, activation=None, kernel_initializer=tf.initializers.variance_scaling)
					if policy_size > 1:
						policy_batch = tf.reshape(policy_batch, [-1,policy_size,policy_depth])
				output_list.append(policy_batch)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return output_list
		
	def _rnn_layer(self, input, scope, name="", share_trainables=True):
		rnn = RNN(type='LSTM', direction=1, units=64, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
		internal_default_state = rnn.default_state()
		layer_type = rnn.type
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			output, internal_final_state = rnn.process_batches(
				input=input, 
				initial_states=internal_initial_state, 
				sizes=self.size_batch
			)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
		