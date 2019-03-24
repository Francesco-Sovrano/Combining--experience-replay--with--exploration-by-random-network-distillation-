# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from agent.network.actor_critic.base_network import Base_Network, is_continuous_control
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAISmall_Network(Base_Network):

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			input = tf.layers.conv2d(name='CNN_Conv1', inputs=input, filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv2', inputs=input, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv3', inputs=input, filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _weights_layer(self, input, weights, scope, name="", share_trainables=True):
		layer_type = 'Weights'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			kernel = tf.stop_gradient(weights['kernel'])
			kernel = tf.transpose(kernel, [1, 0])
			kernel = tf.layers.dense(name='TS_Dense0', inputs=kernel, units=1, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			kernel = tf.reshape(kernel, [1,-1])
			bias = tf.stop_gradient(weights['bias'])
			bias = tf.reshape(bias, [1,-1])
			weight_state = tf.concat((kernel, bias), -1)
			weight_state = tf.layers.dense(name='TS_Dense1', inputs=weight_state, units=64, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			weight_state = tf.reshape(weight_state, [-1])
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
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
		
	def _value_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'Value'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = input + tf.layers.dense(name='Value_Dense1', inputs=input, units=input.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(0.1))
			if self.qvalue_estimation:
				policy_depth = sum(h['depth'] for h in self.policy_heads)
				policy_size = sum(h['size'] for h in self.policy_heads)
				units = policy_size*max(1,policy_depth)
				output = [
					tf.layers.dense(name='Value_Q{}_Dense1'.format(i), inputs=input, units=units, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
					for i in range(self.value_count)
				]
				output = tf.stack(output)
				output = tf.transpose(output, [1, 0, 2])
				if policy_size > 1 and policy_depth > 1:
					output = tf.reshape(output, [-1,self.value_count,policy_size,policy_depth])
			else:
				output = [ # Keep value heads separated
					tf.layers.dense(name='Value_V{}_Dense1'.format(i), inputs=input, units=1, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
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
			input = input + tf.layers.dense(name='Policy_Dense1', inputs=input, units=input.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(0.1))
			output_list = []
			for h,policy_head in enumerate(self.policy_heads):
				policy_depth = policy_head['depth']
				policy_size = policy_head['size']
				if is_continuous_control(policy_depth):
					# build mean
					mu = tf.layers.dense(name='Policy_Mu_Dense{}'.format(h), inputs=input, units=policy_size, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01)) # in (-inf,inf)
					# build standard deviation
					sigma = tf.layers.dense(name='Policy_Sigma_Dense{}'.format(h), inputs=input, units=policy_size, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01)) # in (-inf,inf)
					# clip mu and sigma to avoid numerical instabilities
					clipped_mu = tf.clip_by_value(mu, -1,1) # in [-1,1]
					clipped_sigma = tf.clip_by_value(tf.abs(sigma), 1e-4,1) # in [1e-4,1] # sigma must be greater than 0
					# build policy batch
					policy_batch = tf.stack([clipped_mu, clipped_sigma])
					policy_batch = tf.transpose(policy_batch, [1, 0, 2])
				else: # discrete control
					policy_batch = tf.layers.dense(name='Policy_Logits_Dense{}'.format(h), inputs=input, units=policy_size*policy_depth, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
					if policy_size > 1:
						policy_batch = tf.reshape(policy_batch, [-1,policy_size,policy_depth])
				output_list.append(policy_batch)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return output_list
		
	def _rnn_layer(self, input, scope, name="", share_trainables=True):
		rnn = RNN(type='GRU', direction=1, units=256, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
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
			return output, (internal_initial_state,internal_default_state,internal_final_state)
		