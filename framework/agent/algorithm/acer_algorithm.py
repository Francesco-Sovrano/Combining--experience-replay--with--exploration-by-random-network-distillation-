# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from agent.algorithm.ac_algorithm import AC_Algorithm, merge_splitted_advantages

import options
flags = options.get()

from agent.algorithm.loss.policy_loss import PolicyLoss
from agent.algorithm.loss.value_loss import ValueLoss
from agent.network import *
from utils.misc import accumulate

# Wang, Ziyu, et al. "Sample efficient actor-critic with experience replay." arXiv preprint arXiv:1611.01224 (2016). APA	
class ACER_Algorithm(AC_Algorithm):
	replay_critic = True
	importance_weight_clipping_factor = 10.
	
	@staticmethod
	def get_reversed_cumulative_return(gamma, last_value, reversed_reward, reversed_value, reversed_extra):
		def retrace(gamma, last_value, reversed_reward, reversed_value, reversed_qvalue, reversed_policy_ratio):
			def get_return(state_value_retraced_after, reward, state_value, policy_ratio, action_state_value):
				state_value_retraced_now = reward + gamma*state_value_retraced_after
				state_value_retraced_after = state_value + min(1.,policy_ratio)*(state_value_retraced_now - action_state_value)
				return state_value_retraced_now, state_value_retraced_after
			reversed_cumulative_return, _ = zip(*accumulate(
				iterable=zip(reversed_reward, reversed_value, reversed_policy_ratio, reversed_qvalue), 
				func=lambda cumulative_value,reward_v_ratio_q: get_return(
					state_value_retraced_after=cumulative_value[1],
					reward=reward_v_ratio_q[0],
					state_value=reward_v_ratio_q[1],
					policy_ratio=reward_v_ratio_q[2],
					action_state_value=reward_v_ratio_q[3]
				),
				initial_value=(0.,last_value) # initial cumulative_value
			))
			return reversed_cumulative_return
		reversed_policy_ratio, reversed_qvalue = zip(*reversed_extra)
		return retrace(
			gamma=gamma, 
			last_value=last_value, 
			reversed_reward=reversed_reward, 
			reversed_value=reversed_value, 
			reversed_qvalue=reversed_qvalue, 
			reversed_policy_ratio=reversed_policy_ratio
		)
	
	def initialize_network(self, qvalue_estimation=False):
		super().initialize_network(qvalue_estimation=True)
			
	def _get_policy_loss(self, builder):
		# bugged!
		old_mean_action = self.old_policy_distributions.mean()
		old_mean_cross_entropy = self.old_policy_distributions.cross_entropy(old_mean_action)
		new_mean_action = self.new_policy_distributions.mean()
		new_mean_cross_entropy = self.new_policy_distributions.cross_entropy(new_mean_action)
	# Truncated importance sampling
		# Build variables used for retracing -> retrace outside graph and then feed cumulative_reward_batch to graph
		self.ratio_batch = builder.get_ratio()
		if self.is_continuous_control():
			mean_action = tf.exp(-new_mean_cross_entropy)
			chosen_action = tf.exp(-self.new_cross_entropy_sample)
			value_reduction_axis = -1
		else:
			mean_action = new_mean_action
			chosen_action = self.old_action_batch
			value_reduction_axis = -1 if self.policy_size == 1 else [-2,-1]
		# Add extra dimension to actions for multiplicating them with critic batches that may have multiple heads
		mean_action = tf.expand_dims(mean_action, axis=1)
		chosen_action = tf.expand_dims(chosen_action, axis=1)
		# State value
		self.state_value_batch = tf.reduce_sum(mean_action*self.critic_batch, axis=value_reduction_axis)
		# Action-State value
		self.action_state_value_batch = tf.reduce_sum(chosen_action*self.critic_batch, axis=value_reduction_axis)
		# Build advantage using retraced cumulative_reward_batch
		advantage_batch = tf.minimum(self.importance_weight_clipping_factor, self.ratio_batch) * self.advantage_batch
		loss_truncated_importance = builder.get(advantage_batch)	
	# Bias correction for the truncation
		new_mean_cross_entropy = tf.expand_dims(new_mean_cross_entropy,-1)
		old_mean_cross_entropy = tf.expand_dims(old_mean_cross_entropy,-1)
		if self.policy_size == 1:
			new_mean_cross_entropy = tf.expand_dims(new_mean_cross_entropy,1)
			old_mean_cross_entropy = tf.expand_dims(old_mean_cross_entropy,1)
		builder_bc = PolicyLoss(
			global_step=self.global_step,
			type=flags.policy_loss, 
			cross_entropy=new_mean_cross_entropy, 
			old_cross_entropy=old_mean_cross_entropy
		)
		value_batch = tf.reduce_sum(self.critic_batch, axis=-2) if self.policy_size > 1 else self.critic_batch
		advantage_batch_bc = value_batch - tf.expand_dims(self.state_value_batch, axis=-1)
		# Merge intrisic and extrinsic rewards
		if self.value_count > 1:
			advantage_batch_bc = tf.map_fn(fn=merge_splitted_advantages, elems=advantage_batch_bc)
		# Clip
		advantage_batch_bc *= tf.nn.relu(1.0 - self.importance_weight_clipping_factor / builder_bc.get_ratio())
		loss_bias_correction = builder_bc.get(advantage_batch_bc)
	# Total policy loss
		return loss_truncated_importance + loss_bias_correction
		
	def predict_value(self, value_dict):
		value_batch, bootstrap_value, _ = super().predict_value(value_dict)
		# Get value batch
		feed_dict={
			self.state_batch: value_dict['states'],
			self.old_policy_batch: value_dict['policies'],
			self.old_action_batch: value_dict['actions']
		}
		feed_dict.update(self._get_internal_state_feed(value_dict['internal_state']))
		if self.concat_size > 0:
			feed_dict.update({self.state_concat_batch : value_dict['concats']})
		extra_batch = tf.get_default_session().run(fetches=[self.ratio_batch, self.action_state_value_batch], feed_dict=feed_dict)
		extra_batch = tuple(zip(*extra_batch))
		return value_batch, bootstrap_value, extra_batch
