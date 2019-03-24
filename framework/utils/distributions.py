# -*- coding: utf-8 -*-
import tensorflow as tf

class Categorical(object):
	
	def __init__(self, logits):
		self.logits = logits
		self.logits_shape = self.logits.get_shape().as_list()
		
	def is_multi_categorical(self):
		return len(self.logits_shape) > 2
		
	def mean(self):
		return tf.contrib.layers.softmax(self.logits) # automatically handles the multi-categorical case
	
	def mode(self):
		return tf.argmax(self.logits, axis=-1)
	
	def kl_divergence(self, other): # https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
		assert isinstance(other, Categorical)
		a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
		a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
		ea0 = tf.exp(a0)
		ea1 = tf.exp(a1)
		z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
		z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
		p0 = ea0 / z0
		return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
	
	def cross_entropy(self, samples):
		return tf.nn.softmax_cross_entropy_with_logits_v2(labels=samples, logits=self.logits)

	def entropy(self):
		scaled_logits = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
		exp_scaled_logits = tf.exp(scaled_logits)
		sum_exp_scaled_logits = tf.reduce_sum(exp_scaled_logits, axis=-1, keepdims=True)
		avg_exp_scaled_logits = exp_scaled_logits / sum_exp_scaled_logits
		return tf.reduce_sum(avg_exp_scaled_logits * (tf.log(sum_exp_scaled_logits) - scaled_logits), axis=-1)
			
	def sample(self, one_hot=True):
		depth = self.logits_shape[-1] # depth of the one hot vector
		if self.is_multi_categorical(): # multi-categorical sampling
			logits = tf.reshape(self.logits, [-1,depth])
		else:
			logits = self.logits
		u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
		samples = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
		if self.is_multi_categorical(): # multi-categorical sampling
			samples = tf.reshape(samples, [-1,self.logits_shape[-2]])
		return self.get_sample_one_hot(samples) if one_hot else samples
	
	def get_sample_one_hot(self, samples):
		depth = self.logits_shape[-1] # depth of the one hot vector
		return tf.one_hot(indices=samples, depth=depth, dtype=tf.uint8)
		
class Normal(object):

	def __init__(self, mean, std):
		self.mu = mean
		self.std = std
		self.distribution = tf.distributions.Normal(mean, std, validate_args=False) # validate_args is computationally more expensive
	
	def mean(self):
		return self.mu
	
	def mode(self):
		return self.mu
	
	def kl_divergence(self, other): # https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
		assert isinstance(other, Normal)
		return self.distribution.kl_divergence(other.distribution)
	
	def cross_entropy(self, samples):
		return -self.distribution.log_prob(samples) # probability density function

	def entropy(self):
		return self.distribution.entropy()
			
	def sample(self):
		return self.distribution.sample()