# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.tensorflow_utils as tf_utils

import options
flags = options.get()

class ValueLoss(object):
	def __init__(self, global_step, type, estimation, old_estimation, cumulative_reward):
		self.global_step = global_step
		self.type = type.lower()
		self.estimation = estimation
		# Stop gradients
		self.old_estimation = tf.stop_gradient(old_estimation)
		self.cumulative_return = tf.stop_gradient(cumulative_reward)
		# Get reduce function
		self.reduce_function = eval('tf.reduce_{}'.format(flags.loss_type))
		
	def get(self):
		return eval('self.{}'.format(self.type))()
			
	def vanilla(self):
		# reduce over batches (1st ax)
		losses = self.reduce_function(tf.squared_difference(self.cumulative_return, self.estimation), 0)
		# sum values (last ax)
		return 0.5*tf.reduce_sum(losses)
				
	# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
	def pvo(self):
		# clip
		clip_range = tf_utils.get_annealable_variable(
			function_name=flags.clip_annealing_function, 
			initial_value=flags.clip, 
			global_step=self.global_step, 
			decay_steps=flags.clip_decay_steps, 
			decay_rate=flags.clip_decay_rate
		) if flags.clip_decay else flags.clip
		clip_range = tf.cast(clip_range, self.estimation.dtype)
		# clipped estimation
		estimation_clipped = self.old_estimation + tf.clip_by_value(self.estimation-self.old_estimation, -clip_range, clip_range)
		max = tf.maximum(tf.abs(self.cumulative_return-self.estimation),tf.abs(self.cumulative_return-estimation_clipped))
		# reduce over batches (1st ax)
		losses = self.reduce_function(tf.square(max), 0)
		# sum values (last ax)
		return 0.5*tf.reduce_sum(losses)
