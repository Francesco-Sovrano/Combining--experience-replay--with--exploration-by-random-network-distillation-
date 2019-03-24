# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow as tf
from agent.network.actor_critic.base_network import Base_Network
import utils.tensorflow_utils as tf_utils

class Impala_Network(Base_Network):
	"""
	Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
	Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
	"""
	dropout_probability = 0.
	use_batch_norm = False
	# depths = [32, 64, 64, 64, 64] # Large
	depths=[16, 32, 32] # Small

	def conv_layer(self, out, depth):
		out = tf.layers.conv2d(out, depth, 3, padding='same')
		if self.dropout_probability > 0:
			out = tf.layers.dropout(inputs=out, rate=self.dropout_probability)
		if self.use_batch_norm:
			out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)
		return out

	def residual_block(self, inputs):
		depth = inputs.get_shape()[-1].value
		out = tf.nn.relu(inputs)
		out = self.conv_layer(out, depth)
		out = tf.nn.relu(out)
		out = self.conv_layer(out, depth)
		return out + inputs

	def conv_sequence(self, inputs, depth):
		out = self.conv_layer(inputs, depth)
		out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
		out = self.residual_block(out)
		out = self.residual_block(out)
		return out

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			for depth in self.depths:
				input = self.conv_sequence(input, depth)
			input = tf.layers.flatten(input)
			input = tf.nn.relu(input)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=256, activation=tf.nn.relu)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
