# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
from utils.misc import is_tuple, decompress, compress
from agent.algorithm.ac_algorithm import merge_splitted_advantages
import options
flags = options.get()

class ExperienceBatch():
	# Use slots to save memory, because this type of object may be spawned very frequently
	__slots__ = (
		'_model_size', '_batch_keys', '_agent_list', '_agent_position_list',
		'bootstrap', 'terminal',
		'states', 'new_states', 
		'internal_states', 'new_internal_states', 
		'actions', 'action_masks', 'policies', 'values',
		'rewards', 'manipulated_rewards', 
		'cumulative_returns', 'advantages', 'extras'
	)
	
	def __init__(self, model_size):
		self._model_size = model_size
		self._agent_list = tuple(range(self._model_size))
		self._batch_keys = []
		# Add batch keys, after setting model size
		self._add_batch_key_if_not_exist([
			'states','new_states',
			'internal_states','new_internal_states',
			'actions','action_masks','policies','values',
			'rewards','manipulated_rewards',
			'cumulative_returns','advantages','extras'
		])
		self._agent_position_list = []
		self.bootstrap = {}
		self.terminal = False
		
	def _add_batch_key_if_not_exist(self, key_list):
		if not is_tuple(key_list):
			key_list = [key_list]
		for key in key_list:
			if key in self._batch_keys:
				continue
			setattr(self,key,[[] for _ in range(self._model_size)]) # do NOT use [[]]*_model_size  
			self._batch_keys.append(key)
		
	def is_empty(self, agents=None):
		if agents is None:
			return not self._agent_position_list
		for agent in agents:
			if len(self.states[agent]) > 0:
				return False
		return True
		
	def get_action(self, agent, position, actions=None, as_dict=False):
		if actions is None:
			actions = self._batch_keys
		if not as_dict:
			return [getattr(self,key)[agent][position] for key in actions]
		return {key:getattr(self,key)[agent][position] for key in actions} 
	
	def has_actions(self, actions=None, agents=None):
		if actions is None: # All actions
			actions = self._batch_keys
		else: # Check if (action) keys are valid
			if not is_tuple(actions):
				actions = (actions,)
			for action in actions:
				if action not in self._batch_keys:
					return False
		if agents is None: # All agents
			agents = self._agent_list
		# Check whether there is at least one action (for every key) inside the batch
		for action in actions:
			for agent in agents:
				action_list = getattr(self,action)
				exist = len(action_list) > agent and len(action_list[agent]) > 0
				if not exist:
					return False
		return True
	
	def get_all_actions(self, actions=None, agents=None, reverse=False):
		if actions is None: # All actions
			actions = self._batch_keys
		if agents is None: # All agents
			agents = self._agent_list
		if len(agents) == 1: # Single agent
			iteration_function = (lambda x: x) if not reverse else (lambda x: tuple(reversed(x)))
			agent = agents[0]
			return tuple(iteration_function(getattr(self,key)[agent]) for key in actions)
		# Not single agent
		result = (
			self.get_action(actions=actions, agent=agent_id, position=pos) 
			for (agent_id,pos) in self.step_generator(agents, reverse=reverse)
		)
		if len(actions) > 1: # More than 1 action
			result = zip(*result)
		return tuple(result)
		
	def set_action(self, feed_dict, agent, position):
		for (key, value) in feed_dict.items():
			q = getattr(self,key)[agent]
			if len(q) <= position: # Add missing steps
				q += [None]*(position-len(q)+1)
			q[position] = value

	def add_action(self, feed_dict, agent_id):
		for (key, value) in feed_dict.items():
			getattr(self,key)[agent_id].append(value)
		# (agent_id, batch_position), for step_generator 
		self._agent_position_list.append( (agent_id, len(self.states[agent_id])-1) )
		
	def get_cumulative_reward(self, agents=None):
		# All agents
		if agents is None or len(agents) == self._model_size:
			return sum( sum(rewards) for rewards in self.rewards )
		# Not all agents
		return sum( sum(rewards) for agent,rewards in enumerate(self.rewards) if agent in agents )
		
	def get_size(self, agents=None):
		# All agents
		if agents is None or len(agents) == self._model_size:
			return sum(len(s) for s in self.states)
		# Not all agents
		return sum(len(s) for agent,s in enumerate(self.states) if agent in agents)

	def step_generator(self, agents=None, reverse=False):
		iteration_function = (lambda x: x) if not reverse else reversed
		# All agents
		if agents is None or len(agents) == self._model_size:
			return iteration_function(self._agent_position_list)
		# Single agent, not all agents
		if len(agents)==1:
			agent = agents[0]
			return ((agent,pos) for pos in iteration_function(range(self.get_size(agents))))
		# Nor single agent, nor all agents
		return ((agent,pos) for (agent,pos) in iteration_function(self._agent_position_list) if agent in agents)
		
	def compute_discounted_cumulative_reward(self, agents, gamma, cumulative_return_builder):
		# Bootstrap
		terminal = self.terminal
		(last_agent,_) = next(self.step_generator(agents, reverse=True))
		# assert last_agent in self.bootstrap, "Must bootstrap next value before computing the discounted cumulative reward"
		last_value = 0 + self.bootstrap[last_agent] # copy by value, summing zero
		# Get accumulation sequences and initial values
		reversed_reward, reversed_value = self.get_all_actions(actions=['manipulated_rewards','values'], agents=agents, reverse=True)
		if flags.split_values: # There are 2 value heads: one for intrinsinc and one for extrinsic rewards
			gamma = np.array([gamma, flags.intrinsic_reward_gamma])
			if terminal: # episodic reward
				if flags.episodic_extrinsic_reward:
					last_value[0] = 0.
				if flags.episodic_intrinsic_reward:
					last_value[1] = 0.
		else:
			# Sum intrinsic and extrinsic rewards
			reversed_reward = list(map(lambda reward: np.sum(reward), reversed_reward))
			if terminal and flags.episodic_extrinsic_reward: # episodic reward
				last_value = 0.
		# Get reversed cumulative return
		reversed_cumulative_return = cumulative_return_builder(
			gamma=gamma, 
			last_value=last_value, 
			reversed_reward=reversed_reward, 
			reversed_value=reversed_value, 
			reversed_extra=self.get_all_actions(actions=['extras'], agents=agents, reverse=True)[0] if self.has_actions(actions=['extras'], agents=agents) else None 
		)
		# Get advantage
		if not flags.runtime_advantage:
			if flags.split_values: # merge intrisic and extrinsic rewards
				reversed_advantage = list(map(lambda ret, val: merge_splitted_advantages(ret-val), reversed_cumulative_return, reversed_value))
			else:
				reversed_advantage = list(map(lambda ret, val: ret-val, reversed_cumulative_return, reversed_value))
		# Add accumulations to batch
		if len(agents) == 1: # Optimized code for single agent
			agent_id = agents[0]
			self.cumulative_returns[agent_id] = list(reversed(reversed_cumulative_return))
			if not flags.runtime_advantage:
				self.advantages[agent_id] = list(reversed(reversed_advantage))
		else: # Multi-agents code
			for i,agent_pos in enumerate(self.step_generator(agents, reverse=True)):
				feed_dict = {'cumulative_returns': reversed_cumulative_return[i]}
				if not flags.runtime_advantage:
					feed_dict['advantages'] = reversed_advantage[i] 
				self.set_action(feed_dict, *agent_pos)
			
	def append(self, batch):
		# Append input batch to current batch (self)
		assert self._model_size<=batch._model_size, "Trying to append a batch with model_size {} to a batch with model_size {}".format(batch._model_size, self._model_size)
		#=======================================================================
		# print('before', self.get_cumulative_reward([0]), batch.get_cumulative_reward([0]))
		#=======================================================================
		pos_base = [len(self.states[agent]) for agent in range(self._model_size)]
		self._agent_position_list += [(agent, pos+pos_base[agent]) for (agent,pos) in batch._agent_position_list]
		for i in range(self._model_size):
			for key in self._batch_keys:
				getattr(self,key)[i].extend(getattr(batch,key)[i]) 
		#=======================================================================
		# print('after', self.get_cumulative_reward([0]), batch.get_cumulative_reward([0]))
		#=======================================================================
		self.bootstrap = batch.bootstrap
		self.terminal = batch.terminal
		
	def compress(self):
		for key in self._batch_keys:
			setattr(self, key, compress(getattr(self,key)))
			
	def decompress(self):
		for key in self._batch_keys:
			setattr(self, key, decompress(getattr(self,key)))

class CompositeBatch():
	def __init__(self, maxlen=None):
		self.maxlen = maxlen
		self._batch_list = deque(maxlen=self.maxlen)
		
	def add(self, batch):
		# Can add only ExperienceBatch items
		assert isinstance(batch, ExperienceBatch)
		self._batch_list.append(batch)
		
	def get(self):
		return self._batch_list
	
	def size(self):
		return len(self._batch_list)
	
	def clear(self):
		if self.maxlen > 1:
			self._batch_list = deque(maxlen=self.maxlen)
	
	def compute_discounted_cumulative_reward(self, agents, gamma, cumulative_return_builder):
		last_batch = self._batch_list[-1]
		(last_agent,_) = next(last_batch.step_generator(agents, reverse=True))
		last_value = last_batch.bootstrap[last_agent]
		for batch in reversed(self._batch_list):
			# Update bootstrap value
			(last_agent,_) = next(batch.step_generator(agents, reverse=True))
			batch.bootstrap[last_agent] = last_value
			# Compute cumulative return
			batch.compute_discounted_cumulative_reward(agents, gamma, cumulative_return_builder)
			# Get next bootstrap value
			(first_agent,_) = next(batch.step_generator(agents, reverse=False))
			last_value = batch.cumulative_returns[first_agent][0]
