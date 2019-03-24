# -*- coding: utf-8 -*-
import tensorflow as tf
from agent.network.actor_critic.base_network import Base_Network

class Towers_Network(Base_Network):
	
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		depth = 2
		input_shape = input.get_shape().as_list()
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			with tf.variable_scope("tower_1"):
				tower1 = tf.layers.conv2d(name='CNN_Tower1_Conv1', inputs=input, activation=tf.nn.relu, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower1 = tf.layers.conv2d(name='CNN_Tower1_Conv2', inputs=tower1, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower1 = tf.layers.max_pooling2d(name='CNN_Tower1_MaxPool1', inputs=tower1, pool_size=(input_shape[1], input_shape[2]), strides=(input_shape[1], input_shape[2]))
				tower1 = tf.layers.flatten(tower1)
			with tf.variable_scope("tower_2"):
				tower2 = tf.layers.max_pooling2d(name='CNN_Tower2_MaxPool1', inputs=input, pool_size=(2, 2), strides=(2, 2), padding='SAME')
				for i in range(depth):
					tower2 = tf.layers.conv2d(name='CNN_Tower2_Conv{}'.format(i), inputs=tower2, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower2 = tf.layers.max_pooling2d(name='CNN_Tower2_MaxPool2', inputs=tower2, pool_size=(max(1,input_shape[1]//2), max(1,input_shape[2]//2)), strides=(max(1,input_shape[1]//2), max(1,input_shape[2]//2)))
				tower2 = tf.layers.flatten(tower2)
			with tf.variable_scope("tower_3"):
				tower3 = tf.layers.max_pooling2d(name='CNN_Tower3_MaxPool1', inputs=input, pool_size=(4, 4), strides=(4, 4), padding='SAME')
				for i in range(depth):
					tower3 = tf.layers.conv2d(name='CNN_Tower3_Conv{}'.format(i), inputs=tower3, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower3 = tf.layers.max_pooling2d(name='CNN_Tower3_MaxPool2', inputs=tower3, pool_size=(max(1,input_shape[1]//4), max(1,input_shape[2]//4)), strides=(max(1,input_shape[1]//4), max(1,input_shape[2]//4)))
				tower3 = tf.layers.flatten(tower3)
			concat = tf.concat([tower1, tower2, tower3], axis=-1)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return concat