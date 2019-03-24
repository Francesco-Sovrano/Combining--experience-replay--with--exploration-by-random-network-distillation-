# -*- coding: utf-8 -*-
import tensorflow as tf
from agent.network.actor_critic.base_network import Base_Network

class SA_Network(Base_Network):
	lstm_units = 128 # the number of units of the LSTM
	
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.conv2d(name='CNN_Conv1', inputs=input, filters=16, kernel_size=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
	
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=128, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.contrib.layers.maxout(inputs=input, num_units=64, axis=-1)
			input = tf.reshape(input, [-1, 64])
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
