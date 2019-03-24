# -*- coding: utf-8 -*-
import utils.tensorflow_utils as tf_utils
import tensorflow as tf
import numpy as np
import itertools as it
from agent.algorithm.loss.policy_loss import PolicyLoss
from agent.algorithm.loss.value_loss import ValueLoss
from utils.distributions import Categorical, Normal
from agent.network import *
from utils.misc import accumulate
from collections import deque
from utils.statistics import Statistics
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

def merge_splitted_advantages(advantage):
	return flags.extrinsic_coefficient*advantage[0] + flags.intrinsic_coefficient*advantage[1]

class AC_Algorithm(object):
	replay_critic = flags.use_GAE
	
	@staticmethod
	def get_reversed_cumulative_return(gamma, last_value, reversed_reward, reversed_value, reversed_extra):
		# GAE
		if flags.use_GAE:
			# Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			def generalized_advantage_estimator(gamma, lambd, last_value, reversed_reward, reversed_value):
				# AC_Algorithm.replay_critic = True
				def get_return(last_gae, last_value, reward, value):
					new_gae = reward + gamma*last_value - value + gamma*lambd*last_gae
					return new_gae, value
				reversed_cumulative_advantage, _ = zip(*accumulate(
					iterable=zip(reversed_reward, reversed_value), 
					func=lambda cumulative_value,reward_value: get_return(
						last_gae=cumulative_value[0], 
						last_value=cumulative_value[1], 
						reward=reward_value[0], 
						value=reward_value[1]
					),
					initial_value=(0.,last_value) # initial cumulative_value
				))
				reversed_cumulative_return = tuple(map(lambda adv,val: adv+val, reversed_cumulative_advantage, reversed_value))
				return reversed_cumulative_return
			return generalized_advantage_estimator(
				gamma=gamma, 
				lambd=flags.lambd, 
				last_value=last_value, 
				reversed_reward=reversed_reward, 
				reversed_value=reversed_value
			)
		# Vanilla discounted cumulative reward
		else:
			def vanilla(gamma, last_value, reversed_reward):
				def get_return(last_return, reward):
					return reward + gamma*last_return
				reversed_cumulative_return = tuple(accumulate(
					iterable=reversed_reward, 
					func=lambda cumulative_value,reward: get_return(last_return=cumulative_value, reward=reward),
					initial_value=last_value # initial cumulative_value
				))
				return reversed_cumulative_return
			return vanilla(
				gamma=gamma, 
				last_value=last_value, 
				reversed_reward=reversed_reward
			)
	
	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None):
		self.parameters_type = eval('tf.{}'.format(flags.parameters_type))
		self.beta = beta if beta is not None else flags.beta
		self.value_count = 2 if flags.split_values else 1
		# initialize
		self.training = training
		self.group_id = group_id
		self.model_id = model_id
		self.id = '{0}_{1}'.format(self.group_id,self.model_id) # model id
		self.parent = parent if parent is not None else self # used for sharing with other models in hierarchy, if any
		self.sibling = sibling if sibling is not None else self # used for sharing with other models in hierarchy, if any
		# Environment info
		action_shape = environment_info['action_shape']
		self.policy_heads = [
			{
				'size':head[0], # number of actions to take
				'depth':head[1] if len(head) > 1 else 0 # number of discrete action types: set 0 for continuous control
			}
			for head in action_shape
		]
		state_shape = environment_info['state_shape']
		self.state_heads = [
			{'shape':head}
			for head in state_shape
		]
		self.state_scaler = environment_info['state_scaler'] # state scaler, for saving memory (eg. in case of RGB input: uint8 takes less memory than float64)
		self.has_masked_actions = environment_info['has_masked_actions']
		# Create the network
		self.build_input_placeholders()
		self.initialize_network()
		self.build_network()
		# Stuff for building the big-batch and optimize training computations
		self._big_batch_feed = [{},{}]
		self._batch_count = [0,0]
		self._train_batch_size = flags.batch_size*flags.big_batch_size
		# Statistics
		self._train_statistics = Statistics(flags.episode_count_for_evaluation)
		#=======================================================================
		# self.loss_distribution_estimator = RunningMeanStd(batch_size=flags.batch_size)
		#=======================================================================
		self.actor_loss_is_too_small = False
		
	def get_statistics(self):
		return self._train_statistics.get()
	
	def build_input_placeholders(self):
		print( "Building network {} input placeholders".format(self.id) )
		self.constrain_replay = flags.constraining_replay and flags.replay_mean > 0
		self.is_replayed_batch = self._scalar_placeholder(dtype=tf.bool, batch_size=1, name="replay")
		self.state_mean_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_mean{}".format(i)) for i,head in enumerate(self.state_heads)] 
		self.state_std_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_std{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.state_batch = [self._state_placeholder(shape=head['shape'], name="state{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.size_batch = self._scalar_placeholder(dtype=tf.int32, name="size")
		for i,state in enumerate(self.state_batch):
			print( "	[{}]State{} shape: {}".format(self.id, i, state.get_shape()) )
		self.reward_batch = self._value_placeholder("reward")
		print( "	[{}]Reward shape: {}".format(self.id, self.reward_batch.get_shape()) )
		self.cumulative_return_batch = self._value_placeholder("cumulative_return")
		print( "	[{}]Cumulative Return shape: {}".format(self.id, self.cumulative_return_batch.get_shape()) )
		if not flags.runtime_advantage:
			self.advantage_batch = self._scalar_placeholder("advantage")
			print( "	[{}]Advantage shape: {}".format(self.id, self.advantage_batch.get_shape()) )
		self.old_state_value_batch = self._value_placeholder("old_state_value")
		self.old_policy_batch = [self._policy_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_policy{}".format(i)) for i,head in enumerate(self.policy_heads)]
		self.old_action_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_action_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
		if self.has_masked_actions:
			self.old_action_mask_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=1, name="old_action_mask_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
			
	def _policy_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		if is_continuous_control(policy_depth):
			shape = [batch_size,2,policy_size]
		else: # Discrete control
			shape = [batch_size,policy_size,policy_depth] if policy_size > 1 else [batch_size,policy_depth]
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)
			
	def _action_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		shape = [batch_size]
		if policy_size > 1 or is_continuous_control(policy_depth):
			shape.append(policy_size)
		if policy_depth > 1:
			shape.append(policy_depth)
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)
		
	def _value_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=self.parameters_type, shape=[batch_size,self.value_count], name=name)
	
	def _scalar_placeholder(self, name=None, batch_size=None, dtype=None):
		if dtype is None:
			dtype=self.parameters_type
		return tf.placeholder(dtype=dtype, shape=[batch_size], name=name)
		
	def _state_placeholder(self, shape, name=None, batch_size=None):
		shape = [batch_size] + list(shape)
		input = tf.zeros(shape if batch_size is not None else [1] + shape[1:], dtype=self.parameters_type) # default value
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it
		
	def build_optimizer(self, optimization_algoritmh):
		# global step
		global_step = tf.Variable(0, trainable=False)
		# learning rate
		learning_rate = tf_utils.get_annealable_variable(
			function_name=flags.alpha_annealing_function, 
			initial_value=flags.alpha, 
			global_step=global_step, 
			decay_steps=flags.alpha_decay_steps, 
			decay_rate=flags.alpha_decay_rate
		) if flags.alpha_decay else flags.alpha
		# gradient optimizer
		optimizer = {}
		for p in self.get_network_partitions():
			optimizer[p] = tf_utils.get_optimization_function(optimization_algoritmh)(learning_rate=learning_rate, use_locking=True)
		print("Gradient {} optimized by {}".format(self.id, optimization_algoritmh))
		return optimizer, global_step
	
	def get_network_partitions(self):
		return ['Actor','Critic','Reward']	
	
	def initialize_network(self, qvalue_estimation=False):
		self.network = {}
		batch_dict = {
			'state': self.state_batch, 
			'state_mean': self.state_mean_batch,
			'state_std': self.state_std_batch,
			'size': self.size_batch
		}
		# Build intrinsic reward network here because we need its internal state for building actor and critic
		self.network['Reward'] = IntrinsicReward_Network(id=self.id, batch_dict=batch_dict, scope_dict={'self': "IRNet{0}".format(self.id)}, training=self.training)
		if flags.intrinsic_reward:
			reward_network_output = self.network['Reward'].build()
			self.intrinsic_reward_batch = reward_network_output[0]
			self.intrinsic_reward_loss = reward_network_output[1]
			self.training_state = reward_network_output[2]
			print( "	[{}]Intrinsic Reward shape: {}".format(self.id, self.intrinsic_reward_batch.get_shape()) )
			print( "	[{}]Training State Kernel shape: {}".format(self.id, self.training_state['kernel'].get_shape()) )
			print( "	[{}]Training State Bias shape: {}".format(self.id, self.training_state['bias'].get_shape()) )		
			batch_dict['training_state'] = self.training_state
		# Build actor and critic
		for p in ('Actor','Critic'):
			if flags.separate_actor_from_critic: # non-shared graph
				node_id = self.id + p
				parent_id = self.parent.id + p
				sibling_id = self.sibling.id + p
			else: # shared graph
				node_id = self.id
				parent_id = self.parent.id
				sibling_id = self.sibling.id
			scope_dict = {
				'self': "Net{0}".format(node_id),
				'parent': "Net{0}".format(parent_id),
				'sibling': "Net{0}".format(sibling_id)
			}
			self.network[p] = eval('{}_Network'.format(flags.network_configuration))(
				id=node_id, 
				qvalue_estimation=qvalue_estimation,
				policy_heads=self.policy_heads,
				batch_dict=batch_dict,
				scope_dict=scope_dict, 
				training=self.training,
				value_count=self.value_count,
				state_scaler=self.state_scaler
			)
				
	def build_network(self):
		# Actor & Critic
		self.actor_batch, _ = self.network['Actor'].build(name='Actor', has_actor=True, has_critic=False, use_internal_state=flags.network_has_internal_state)
		for i,b in enumerate(self.actor_batch): 
			print( "	[{}]Actor{} output shape: {}".format(self.id, i, b.get_shape()) )
		_, self.critic_batch = self.network['Critic'].build(name='Critic', has_actor=False, has_critic=True, use_internal_state=flags.network_has_internal_state)
		print( "	[{}]Critic output shape: {}".format(self.id, self.critic_batch.get_shape()) )
		# Sample action, after getting keys
		self.action_batch, self.hot_action_batch = self.sample_actions()
		for i,b in enumerate(self.action_batch): 
			print( "	[{}]Action{} output shape: {}".format(self.id, i, b.get_shape()) )
		for i,b in enumerate(self.hot_action_batch): 
			print( "	[{}]HotAction{} output shape: {}".format(self.id, i, b.get_shape()) )
			
	def sample_actions(self):
		action_batch = []
		hot_action_batch = []
		for h,actor_head in enumerate(self.actor_batch):
			if is_continuous_control(self.policy_heads[h]['depth']):
				new_policy_batch = tf.transpose(actor_head, [1, 0, 2])
				sample_batch = Normal(new_policy_batch[0], new_policy_batch[1]).sample()
				action = tf.clip_by_value(sample_batch, -1,1)
				action_batch.append(action) # Sample action batch in forward direction, use old action in backward direction
				hot_action_batch.append(action)
			else: # discrete control
				distribution = Categorical(actor_head)
				action = distribution.sample(one_hot=False) # Sample action batch in forward direction, use old action in backward direction
				action_batch.append(action)
				hot_action_batch.append(distribution.get_sample_one_hot(action))
		# Give self esplicative name to output for easily retrieving it in frozen graph
		# tf.identity(action_batch, name="action")
		return action_batch, hot_action_batch
		
	def _get_policy_loss_builder(self, new_policy_distributions, old_policy_distributions, old_action_batch, old_action_mask_batch=None):
		cross_entropy = new_policy_distributions.cross_entropy(old_action_batch)
		old_cross_entropy = old_policy_distributions.cross_entropy(old_action_batch)
		if old_action_mask_batch is not None:
			# stop gradient computation on masked elements and remove them from loss (zeroing)
			cross_entropy = tf.where(
			    tf.equal(old_action_mask_batch,1),
			    x=cross_entropy, # true branch
			    y=tf.stop_gradient(old_action_mask_batch) # false branch
			)
			old_cross_entropy = tf.where(
			    tf.equal(old_action_mask_batch,1),
			    x=old_cross_entropy, # true branch
			    y=tf.stop_gradient(old_action_mask_batch) # false branch
			)
		return PolicyLoss(
			global_step= self.global_step,
			type= flags.policy_loss,
			cross_entropy= cross_entropy, 
			old_cross_entropy= old_cross_entropy, 
			entropy= new_policy_distributions.entropy(), 
			beta= self.beta
		)
		
	def _get_policy_loss(self, builder):
		if flags.runtime_advantage:
			self.advantage_batch = self.cumulative_return_batch - self.state_value_batch # baseline is always up to date
			if self.value_count > 1:
				self.advantage_batch = tf.map_fn(fn=merge_splitted_advantages, elems=self.advantage_batch) 
		return builder.get(self.advantage_batch)
	
	def _get_value_loss_builder(self):
		return ValueLoss(
			global_step=self.global_step,
			type=flags.value_loss,
			estimation=self.state_value_batch, 
			old_estimation=self.old_state_value_batch, 
			cumulative_reward=self.cumulative_return_batch
		)
		
	def _get_value_loss(self, builder):
		return flags.value_coefficient * builder.get() # usually critic has lower learning rate
		
	def prepare_loss(self, global_step):
		self.global_step = global_step
		print( "Preparing loss {}".format(self.id) )
		self.state_value_batch = self.critic_batch
		# [Policy distribution]
		old_policy_distributions = []
		new_policy_distributions = []
		policy_loss_builder = []
		for h,policy_head in enumerate(self.policy_heads):
			if is_continuous_control(policy_head['depth']):
				# Old policy
				old_policy_batch = tf.transpose(self.old_policy_batch[h], [1, 0, 2])
				old_policy_distributions.append( Normal(old_policy_batch[0], old_policy_batch[1]) )
				# New policy
				new_policy_batch = tf.transpose(self.actor_batch[h], [1, 0, 2])
				new_policy_distributions.append( Normal(new_policy_batch[0], new_policy_batch[1]) )
			else: # discrete control
				old_policy_distributions.append( Categorical(self.old_policy_batch[h]) ) # Old policy
				new_policy_distributions.append( Categorical(self.actor_batch[h]) ) # New policy
			builder = self._get_policy_loss_builder(new_policy_distributions[h], old_policy_distributions[h], self.old_action_batch[h], self.old_action_mask_batch[h] if self.has_masked_actions else None)
			policy_loss_builder.append(builder)
		# [Actor loss]
		self.policy_loss = sum(self._get_policy_loss(b) for b in policy_loss_builder)
		# [Debug variables]
		self.policy_kl_divergence = sum(b.approximate_kullback_leibler_divergence() for b in policy_loss_builder)
		self.policy_clipping_frequency = sum(b.get_clipping_frequency() for b in policy_loss_builder)/len(policy_loss_builder) # take average because clipping frequency must be in [0,1]
		self.policy_entropy_regularization = sum(b.get_entropy_regularization() for b in policy_loss_builder)
		# [Critic loss]
		value_loss_builder = self._get_value_loss_builder()
		self.value_loss = self._get_value_loss(value_loss_builder)
		# [Entropy regularization]
		if flags.entropy_regularization:
			self.policy_loss += -self.policy_entropy_regularization
		# [Constraining Replay]
		if self.constrain_replay:
			constrain_loss = sum(
				0.5*builder.reduce_function(tf.squared_difference(new_distribution.mean(), tf.stop_gradient(old_action))) 
				for builder, new_distribution, old_action in zip(policy_loss_builder, new_policy_distributions, self.old_action_batch)
			)
			self.policy_loss += tf.cond(
				pred=self.is_replayed_batch[0], 
				true_fn=lambda: constrain_loss,
				false_fn=lambda: tf.constant(0., dtype=self.parameters_type)
			)
		# [Total loss]
		self.total_loss = self.policy_loss + self.value_loss
		if flags.intrinsic_reward:
			self.total_loss += self.intrinsic_reward_loss
		
	def get_shared_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(it.chain.from_iterable(self.network[p].shared_keys for p in partitions))
		return sorted(key_list, key=lambda x: x.name)
	
	def get_update_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(it.chain.from_iterable(self.network[p].update_keys for p in partitions))
		return sorted(key_list, key=lambda x: x.name)

	def _get_train_op(self, global_step, optimizer, loss, shared_keys, update_keys, global_keys):
		with tf.control_dependencies(update_keys): # control_dependencies is for batch normalization
			grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=shared_keys)
			# grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
			grad, vars = zip(*grads_and_vars)
			global_grads_and_vars = tuple(zip(grad, global_keys))
			return optimizer.apply_gradients(global_grads_and_vars, global_step=global_step)
		
	def minimize_local_loss(self, optimizer, global_step, global_agent): # minimize loss and apply gradients to global vars.
		actor_optimizer, critic_optimizer, reward_optimizer = optimizer.values()
		self.actor_op = self._get_train_op(
			global_step=global_step,
			optimizer=actor_optimizer, 
			loss=self.policy_loss, 
			shared_keys=self.get_shared_keys(['Actor']), 
			global_keys=global_agent.get_shared_keys(['Actor']),
			update_keys=self.get_update_keys(['Actor'])
		)
		self.critic_op = self._get_train_op(
			global_step=global_step,
			optimizer=critic_optimizer, 
			loss=self.value_loss, 
			shared_keys=self.get_shared_keys(['Critic']), 
			global_keys=global_agent.get_shared_keys(['Critic']),
			update_keys=self.get_update_keys(['Critic'])
		)
		if flags.intrinsic_reward:
			self.reward_op = self._get_train_op(
				global_step=global_step,
				optimizer=reward_optimizer, 
				loss=self.intrinsic_reward_loss, 
				shared_keys=self.get_shared_keys(['Reward']), 
				global_keys=global_agent.get_shared_keys(['Reward']),
				update_keys=self.get_update_keys(['Reward'])
			)
			
	def bind_sync(self, src_network, name=None):
		with tf.name_scope(name, "Sync{0}".format(self.id),[]) as name:
			src_vars = src_network.get_shared_keys()
			dst_vars = self.get_shared_keys()
			sync_ops = []
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var) # no need for locking dst_var
				sync_ops.append(sync_op)
			self.sync_op = tf.group(*sync_ops, name=name)
				
	def sync(self):
		tf.get_default_session().run(fetches=self.sync_op)
		
	def predict_reward(self, reward_dict):
		assert flags.intrinsic_reward, "Cannot get intrinsic reward if the RND layer is not built"
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=reward_dict['states'])
		feed_dict.update( self._get_multihead_feed(target=self.state_mean_batch, source=[reward_dict['state_mean']]) )
		feed_dict.update( self._get_multihead_feed(target=self.state_std_batch, source=[reward_dict['state_std']]) )
		# Return intrinsic_reward
		return tf.get_default_session().run(fetches=self.intrinsic_reward_batch, feed_dict=feed_dict)
				
	def predict_value(self, value_dict):
		state_batch = value_dict['states']
		size_batch = value_dict['sizes']
		bootstrap = value_dict['bootstrap']
		for i,b in enumerate(bootstrap):
			state_batch = state_batch + [b['state']]
			size_batch[i] += 1
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=state_batch)
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed(value_dict['internal_states']) )
			feed_dict.update( {self.size_batch: size_batch} )
		# Return value_batch
		value_batch = tf.get_default_session().run(fetches=self.state_value_batch, feed_dict=feed_dict)
		return value_batch[:-1], value_batch[-1], None
	
	def predict_action(self, action_dict):
		batch_size = action_dict['sizes']
		batch_count = len(batch_size)
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=action_dict['states'])
		# Internal state
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed( action_dict['internal_states'] ) )
			feed_dict.update( {self.size_batch: batch_size} )
		# Return action_batch, policy_batch, new_internal_state
		action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states = tf.get_default_session().run(fetches=[self.action_batch, self.hot_action_batch, self.actor_batch, self.state_value_batch, self._get_internal_state()], feed_dict=feed_dict)
		# Properly format for output the internal state
		if len(new_internal_states) == 0:
			new_internal_states = [new_internal_states]*batch_count
		else:
			new_internal_states = [
				[
					[
						sub_partition_new_internal_state[i]
						for sub_partition_new_internal_state in partition_new_internal_states
					]
					for partition_new_internal_states in new_internal_states
				]
				for i in range(batch_count)
			]
		# Properly format for output: action and policy may have multiple heads, swap 1st and 2nd axis
		action_batch = tuple(zip(*action_batch))
		hot_action_batch = tuple(zip(*hot_action_batch))
		policy_batch = tuple(zip(*policy_batch))
		# Return output
		return action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states
		
	def _get_internal_state(self):
		return tuple(self.network[p].internal_initial_state for p in self.get_network_partitions() if self.network[p].use_internal_state)
	
	def _get_internal_state_feed(self, internal_states):
		if not flags.network_has_internal_state:
			return {}
		feed_dict = {}
		i = 0
		for partition in self.get_network_partitions():
			network_partition = self.network[partition]
			if network_partition.use_internal_state:
				partition_batch_states = [
					network_partition.internal_default_state if internal_state is None else internal_state[i]
					for internal_state in internal_states
				]
				for j, initial_state in enumerate(zip(*partition_batch_states)):
					feed_dict.update( {network_partition.internal_initial_state[j]: initial_state} )
				i += 1
		return feed_dict

	def _get_multihead_feed(self, source, target):
		# Action and policy may have multiple heads, swap 1st and 2nd axis of source with zip*
		return { t:s for t,s in zip(target, zip(*source)) }

	def prepare_train(self, train_dict, replay):
		''' Prepare training batch, then _train once using the biggest possible batch '''
		train_type = 1 if replay else 0
		# Get global feed
		current_global_feed = self._big_batch_feed[train_type]
		# Build local feed
		local_feed = self._build_train_feed(train_dict)
		# Merge feed dictionary
		for key,value in local_feed.items():
			if key not in current_global_feed:
				current_global_feed[key] = deque(maxlen=self._train_batch_size) # Initializing the main_feed_dict 
			current_global_feed[key].extend(value)
		# Increase the number of batches composing the big batch
		self._batch_count[train_type] += 1
		if self._batch_count[train_type]%flags.big_batch_size == 0: # can _train
			# Reset batch counter
			self._batch_count[train_type] = 0
			# Reset big-batch (especially if network_has_internal_state) otherwise when in GPU mode it's more time and memory efficient to not reset the big-batch, in order to keep its size fixed
			self._big_batch_feed[train_type] = {}
			# Train
			return self._train(feed_dict=current_global_feed, replay=replay, state_mean_std=(train_dict['state_mean'],train_dict['state_std']))
		return None
	
	def _train(self, feed_dict, replay=False, state_mean_std=None):
		# Add replay boolean to feed dictionary
		feed_dict.update( {self.is_replayed_batch: [replay]} )
		# Intrinsic Reward
		if flags.intrinsic_reward:
			state_mean, state_std = state_mean_std
			feed_dict.update( self._get_multihead_feed(target=self.state_mean_batch, source=[state_mean]) )
			feed_dict.update( self._get_multihead_feed(target=self.state_std_batch, source=[state_std]) )
		# Build _train fetches
		train_tuple = (self.actor_op, self.critic_op) if not replay or flags.train_critic_when_replaying else (self.actor_op, )
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if flags.intrinsic_reward and not replay:
			train_tuple += (self.reward_op,)
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		if flags.print_loss: # Get loss values for logging
			fetches += [(self.total_loss, self.policy_loss, self.value_loss)]
		else:
			fetches += [()]
		if flags.print_policy_info: # Debug info
			fetches += [(self.policy_kl_divergence, self.policy_clipping_frequency, self.policy_entropy_regularization)]
		else:
			fetches += [()]
		if flags.intrinsic_reward:
			fetches += [(self.intrinsic_reward_loss, )]
		else:
			fetches += [()]
		# Run
		_, loss, policy_info, reward_info = tf.get_default_session().run(fetches=fetches, feed_dict=feed_dict)
		self.sync()
		# Build and return loss dict
		train_info = {}
		if flags.print_loss:
			train_info["loss_total"], train_info["loss_actor"], train_info["loss_critic"] = loss
		if flags.print_policy_info:
			train_info["actor_kl_divergence"], train_info["actor_clipping_frequency"], train_info["actor_entropy"] = policy_info
		if flags.intrinsic_reward:
			train_info["intrinsic_reward_loss"] = reward_info
		# Build loss statistics
		if train_info:
			self._train_statistics.add(stat_dict=train_info, type='train{}_'.format(self.model_id))
		#=======================================================================
		# if self.loss_distribution_estimator.update([abs(train_info['loss_actor'])]):
		# 	self.actor_loss_is_too_small = self.loss_distribution_estimator.mean <= flags.loss_stationarity_range
		#=======================================================================
		return train_info
		
	def _build_train_feed(self, train_dict):
		# State & Cumulative Return & Old Value
		feed_dict = {
			self.cumulative_return_batch: train_dict['cumulative_returns'],
			self.old_state_value_batch: train_dict['values'],
		}
		feed_dict.update( self._get_multihead_feed(target=self.state_batch, source=train_dict['states']) )
		# Advantage
		if not flags.runtime_advantage:
			feed_dict.update( {self.advantage_batch: train_dict['advantages']} )
		# Old Policy & Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=train_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=train_dict['actions']) )
		if self.has_masked_actions:
			feed_dict.update( self._get_multihead_feed(target=self.old_action_mask_batch, source=train_dict['action_masks']) )
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed([train_dict['internal_state']]) )
			feed_dict.update( {self.size_batch: [len(train_dict['cumulative_returns'])]} )
		return feed_dict
