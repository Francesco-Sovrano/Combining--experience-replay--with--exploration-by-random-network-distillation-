# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.tensorflow_utils as tf_utils

import options
flags = options.get()

class PolicyLoss(object):

	def __init__(self, global_step, type, cross_entropy, old_cross_entropy, entropy=0, beta=0):
		self.global_step = global_step
		self.type = type.lower()
		self.zero = tf.constant(0., dtype=cross_entropy.dtype)
		self.one = tf.constant(1., dtype=cross_entropy.dtype)
		# Clip
		self.clip_range = tf_utils.get_annealable_variable(
			function_name=flags.clip_annealing_function, 
			initial_value=flags.clip, 
			global_step=self.global_step, 
			decay_steps=flags.clip_decay_steps, 
			decay_rate=flags.clip_decay_rate
		) if flags.clip_decay else flags.clip
		self.clip_range = tf.cast(self.clip_range, cross_entropy.dtype)
		# Entropy
		self.beta = beta
		self.entropy = tf.maximum(self.zero, entropy) if flags.only_non_negative_entropy else entropy
		self.cross_entropy = tf.maximum(self.zero, cross_entropy) if flags.only_non_negative_entropy else cross_entropy
		self.old_cross_entropy = tf.maximum(self.zero, old_cross_entropy) if flags.only_non_negative_entropy else old_cross_entropy
		# Sum entropies in case the agent has to predict more than one action
		if len(self.cross_entropy.get_shape()) > 1:
			self.cross_entropy = tf.reduce_sum(self.cross_entropy, 1)
		if len(self.old_cross_entropy.get_shape()) > 1:
			self.old_cross_entropy = tf.reduce_sum(self.old_cross_entropy, 1)
		if len(self.entropy.get_shape()) > 1:
			self.entropy = tf.reduce_sum(self.entropy, 1)
		# Stop gradient
		self.old_cross_entropy = tf.stop_gradient(self.old_cross_entropy)
		# Reduction function
		self.reduce_function = eval('tf.reduce_{}'.format(flags.loss_type))
		self.ratio_batch = self.get_ratio()
		
	def get(self, advantage):
		self.advantage = tf.stop_gradient(advantage)
		return eval('self.{}'.format(self.type))()
			
	def approximate_kullback_leibler_divergence(self): # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
		return 0.5 * self.reduce_function(tf.squared_difference(self.old_cross_entropy,self.cross_entropy))
		
	def get_clipping_frequency(self):
		return tf.reduce_mean(tf.to_float(tf.greater(tf.abs(self.ratio_batch - self.one), self.clip_range)))
		
	def get_entropy_regularization(self):
		if self.beta == 0:
			return self.zero
		return self.beta*self.reduce_function(self.entropy)
		
	def get_ratio(self):
		return tf.exp(self.old_cross_entropy - self.cross_entropy)
			
	def vanilla(self):
		gain = self.advantage*self.cross_entropy
		# Reduce over batch and then sum all components
		return tf.reduce_sum(self.reduce_function(gain, 0))
		
	# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
	def ppo(self):
		clipped_ratio = tf.clip_by_value(self.ratio_batch, self.one-self.clip_range, self.one+self.clip_range)
		gain = tf.maximum(-self.advantage*self.ratio_batch, -self.advantage*clipped_ratio)
		# Reduce over batch and then sum all components
		return tf.reduce_sum(self.reduce_function(gain, 0))
	