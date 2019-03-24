# -*- coding: utf-8 -*-
import numpy as np
import random
import options
flags = options.get()

class Environment(object):
	state_scaler = 1
	
	@staticmethod
	def create_environment(env_type, id=0, training=True):
		if env_type == 'rogue':
			from . import rogue_environment
			return rogue_environment.RogueEnvironment(id)
		elif env_type == 'MultipleProteinAlignment':
			from . import multiple_protein_alignment_environment
			return multiple_protein_alignment_environment.MSAEnvironment(id, training)
		elif env_type == 'car_controller':
			from . import car_controller_environment
			return car_controller_environment.CarControllerEnvironment(id)
		elif env_type == 'sentipolc':
			from . import sentipolc_environment
			return sentipolc_environment.SentiPolcEnvironment(id, training)
		else:
			from . import gym_environment
			return gym_environment.GymEnvironment(id, env_type)
		
	def get_concatenation_size(self):
		return sum(map(lambda x: x[0], self.get_action_shape()))+1
		
	# Last Action-Reward: Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).
	def get_concatenation(self):
		if self.last_action is None:
			return np.zeros(self.get_concatenation_size())
		flatten_action = np.concatenate([np.reshape(a,-1) for a in self.last_action], -1)
		return np.concatenate((flatten_action,[self.last_reward]), -1)

	def process(self, action):
		pass

	def reset(self, data_id=None):
		pass

	def stop(self):
		pass
	
	def sample_random_action(self):
		result = []
		for action_shape in self.get_action_shape():
			if len(action_shape) > 1:
				count, size = action_shape
			else:
				count = action_shape[0]
				size = 0
			if size > 0: # categorical sampling
				samples = (np.random.rand(count)*size).astype(np.uint8)
				result.append(samples if count > 1 else samples[0])
			else: # uniform sampling
				result.append([2*random.random()-1 for _ in range(count)])
		return result
		
	def get_state_shape(self):
		pass
	
	def get_test_result(self):
		return None
		
	def get_dataset_size(self):
		return flags.episode_count_for_evaluation
		
	def evaluate_test_results(self, test_result_file):
		pass
		
	def get_screen_shape(self):
		return self.get_state_shape()
	
	def get_info(self):
		return None
	
	def get_screen(self):
		return None
	
	def get_statistics(self):
		return {}
	
	def has_masked_actions(self):
		return False

