# -*- coding: utf-8 -*-
from agent.worker.environment_manager import EnvironmentManager
from utils.statistics import IndexedStatistics
from environment.environment import Environment
from agent.worker.network_manager import NetworkManager
from utils.running_std import RunningMeanStd
from utils.important_information import ImportantInformation
import options
flags = options.get()

class Group(object):
	
	def __init__(self, group_id, environment_count, global_network, training=True):
		self.group_id = group_id
		self.training = training
		# Get environment info
		tmp_environment = Environment.create_environment(env_type=flags.env_type, training=training)
		self.environment_info = {
			'state_shape': tmp_environment.get_state_shape(),
			'action_shape': tmp_environment.get_action_shape(),
			'state_scaler': tmp_environment.state_scaler,
			'has_masked_actions': tmp_environment.has_masked_actions(),
		}
		# Build network_manager
		self.network_manager = NetworkManager(
			group_id=self.group_id,
			environment_info=self.environment_info,
			global_network=global_network,
			training=self.training
		)
		# Build environments
		self.environment_count = environment_count
		self.worker_list = [
			EnvironmentManager(
				model_size=self.network_manager.model_size, 
				environment_id=env_id, 
				group_id=group_id, 
				training=training
			)
			for env_id in range(self.environment_count)
		]
		# State distribution estimator
		self.state_distribution_estimator = [RunningMeanStd(batch_size=flags.batch_size, shape=shape) for shape in self.environment_info['state_shape']]
		self.network_manager.state_mean = [estimator.mean for estimator in self.state_distribution_estimator]
		self.network_manager.state_std = [estimator.std for estimator in self.state_distribution_estimator]
		ImportantInformation(self.state_distribution_estimator, 'state_distribution_estimator{}'.format(self.group_id))
		# Statistics
		self.group_statistics = IndexedStatistics(max_count=self.environment_count, buffer_must_be_full=True)
		self.has_terminal_worker = False
		self.terminated_episodes = 0
		
	def initialize_environments(self, step_count=0):
		step_per_worker = step_count//len(self.worker_list)
		for worker in self.worker_list:
			states = worker.run_random_steps(step_per_worker)
			for i,sub_state in enumerate(zip(*states)):
				self.state_distribution_estimator[i].update(sub_state)
		self.network_manager.state_mean = [estimator.mean for estimator in self.state_distribution_estimator]
		self.network_manager.state_std = [estimator.std for estimator in self.state_distribution_estimator]
		print("Environment group {} initialized".format(self.group_id))

	def stop(self): # stop current episode
		for worker in self.worker_list:
			worker.stop()
		
	def get_statistics(self):
		stats = self.network_manager.get_statistics()
		stats.update(self.group_statistics.get())
		return stats
	
	def _process_step(self, workers):
		if self.training:
			self.network_manager.sync()
		internal_states = [worker.get_internal_states() for worker in workers]
		states = tuple(worker.environment.last_state for worker in workers)
		actions, hot_actions, policies, values, new_internal_states, agents = self.network_manager.predict_action(states, internal_states)
		new_states, extrinsic_rewards, terminals, action_masks = zip(*[worker.environment.process(action) for worker, action in zip(workers, actions)])
		# Update batch state
		for i in range(len(workers)):
			workers[i].update_batch_state(
				terminal= terminals[i], 
				internal_state= new_internal_states[i]
			)
		# Build batch
		if self.training:
			# Update state distribution estimator
			for i, sub_state in enumerate(zip(*new_states)):
				estimator = self.state_distribution_estimator[i]
				if estimator.update(sub_state):
					self.network_manager.state_mean[i] = estimator.mean
					self.network_manager.state_std[i] = estimator.std
			# Apply actions to batches 
			for i in range(len(workers)):
				# Build action dictionary
				action_dict = {
					'states': states[i],
					'new_states': new_states[i],
					'actions': hot_actions[i],
					'action_masks': action_masks[i],
					'values': values[i],
					'policies': policies[i],
					'internal_states': internal_states[i],
					'new_internal_states': new_internal_states[i],
				}
				# Apply action
				workers[i].apply_action_to_batch(
					agent= agents[i], 
					action_dict= action_dict, 
					extrinsic_reward= extrinsic_rewards[i]
				)
	
	def process(self, global_step=0, batch=True, data_id=0):
		# Prepare episode
		for i,worker in enumerate(self.worker_list):
			if worker.terminal:
				worker.prepare_episode(data_id+i)
		# Initialize new batch
		for worker in self.worker_list:
			worker.initialize_new_batch()
		# Sync to global, if not training
		if not self.training:
			self.network_manager.sync()
		# Build batch
		group_step = 0
		step = 0
		valid_workers = self.worker_list
		while step < flags.batch_size and len(valid_workers) > 0:
			group_step += len(valid_workers)
			self._process_step(valid_workers)
			valid_workers = [worker for worker in self.worker_list if not worker.terminal]
			if batch:
				step += 1
		terminated_works = len(self.worker_list) - len(valid_workers)
		# Terminate batch & replay experience
		if self.training:
			for worker in self.worker_list:
				# Replay experience, before finalizing the new batch
				if global_step > flags.replay_step:
					self.network_manager.try_to_replay_experience()
				# Finalize the new batch and train on it
				self.network_manager.finalize_batch(composite_batch=worker.finalize_batch(global_step), global_step=global_step)
				# Log the new batch
				worker.log_batch(global_step, self.network_manager.agents_set)
		# Statistics
		self.terminated_episodes += terminated_works
		self.has_terminal_worker = terminated_works != 0
		for i,worker in enumerate(self.worker_list):
			if worker.terminal:
				self.group_statistics.set(worker.get_statistics(),i)
		return group_step
