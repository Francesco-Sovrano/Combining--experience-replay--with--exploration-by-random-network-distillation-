# -*- coding: utf-8 -*-
import tensorflow as tf

class Network():
	def __init__(self, id, training):
		self.training = training
		self.id = id
		self.use_internal_state = False
		# Initialize keys collections
		self.shared_keys = []
		self.update_keys = []
	
	def _update_keys(self, scope_name, share_trainables):
		if share_trainables:
			self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name)
		
	def _batch_normalization_layer(self, input, scope, name="", share_trainables=True, renorm=False, center=True, scale=True):
		layer_type = 'BatchNorm'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			batch_norm = tf.layers.BatchNormalization(renorm=renorm, center=center, scale=scale) # renorm when minibaches are too small
			norm_input = batch_norm.apply(input, training=self.training)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return batch_norm, norm_input
			
	def _feature_entropy_layer(self, input, scope, name="", share_trainables=True): # feature entropy measures how much the input is uncommon
		layer_type = 'Fentropy'
		batch_norm, _ = self._batch_normalization_layer(input=input, scope=scope, name=layer_type)
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			fentropy = Normal(batch_norm.moving_mean, tf.sqrt(batch_norm.moving_variance)).cross_entropy(input)
			fentropy = tf.layers.flatten(fentropy)
			if len(fentropy.get_shape()) > 1:
				fentropy = tf.reduce_mean(fentropy, axis=-1)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			return fentropy
