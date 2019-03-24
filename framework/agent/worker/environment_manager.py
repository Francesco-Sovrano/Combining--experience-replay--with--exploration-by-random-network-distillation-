# -*- coding: utf-8 -*-
import os
import shutil # for deleting non-empty directories
import zipfile # for zipping files
import logging
import numpy as np
import utils.plots as plt
from collections import deque
from utils.statistics import Statistics
from utils.misc import flatten, softmax
from environment.environment import Environment
from agent.worker.batch import ExperienceBatch, CompositeBatch

# get command line args
import options
flags = options.get()

class EnvironmentManager(object):
	
	def __init__(self, model_size, group_id, environment_id=0, training=True):
		self.model_size = model_size
		self._training = training
		self.environment_id = environment_id
		self.group_id = group_id
		# Build environment
		self.environment = Environment.create_environment(flags.env_type, self.environment_id, self._training)
		self.extrinsic_reward_manipulator = eval(flags.extrinsic_reward_manipulator)
		self.terminal = True
		self._composite_batch = CompositeBatch(maxlen=flags.replay_buffer_size if flags.replay_mean > 0 else 1)
		# Statistics
		self.__client_statistics = Statistics(flags.episode_count_for_evaluation)
		if self._training:
			#logs
			if not os.path.isdir(flags.log_dir + "/performance"):
				os.mkdir(flags.log_dir + "/performance")
			if not os.path.isdir(flags.log_dir + "/episodes"):
				os.mkdir(flags.log_dir + "/episodes")
			formatter = logging.Formatter('%(asctime)s %(message)s')
			# reward logger
			self.__reward_logger = logging.getLogger('reward_{}_{}'.format(self.group_id, self.environment_id))
			hdlr = logging.FileHandler(flags.log_dir + '/performance/reward_{}_{}.log'.format(self.group_id, self.environment_id))
			hdlr.setFormatter(formatter)
			self.__reward_logger.addHandler(hdlr) 
			self.__reward_logger.setLevel(logging.DEBUG)
			self.__max_reward = float("-inf")
		
	def run_random_steps(self, step_count=0):
		state_batch = []
		self.environment.reset()
		for _ in range(step_count):
			new_state, _, terminal, _ = self.environment.process(self.environment.sample_random_action())
			state_batch.append(new_state)
			if terminal:
				self.environment.reset()
		print("Environment {}.{} initialized".format(self.group_id, self.environment_id))
		return state_batch

	def prepare_episode(self, data_id=None): # initialize a new episode
		self.terminal = False
		# Reset environment
		self.environment.reset(data_id)
		# Internal state
		self._last_internal_state = None
		self._batch = None
		# Episode batches
		self._composite_batch.clear()
		# Episode info
		self.__episode_step = []
		self.__episode_info = {
			'tot_reward': 0,
			'tot_manipulated_reward': 0,
			'tot_value': 0,
			'tot_step': 0
		}
		# Frame info
		if flags.show_episodes == 'none':
			self.save_frame_info = False
		else:
			self.save_frame_info = flags.show_episodes != 'random' or np.random.random() <= flags.show_episode_probability

	def stop(self): # stop current episode
		self.environment.stop()
		
	def print_frames(self, frames, episode_directory):
		print_frame = False
		if flags.show_episodes == 'best':
			if self.__episode_info['tot_reward'] > self.__max_reward:
				self.__max_reward = self.__episode_info['tot_reward']
				print_frame = True
		elif self.save_frame_info:
			print_frame = True
		if not print_frame:
			return
		frames_count = len(frames)
		if frames_count < 1:
			return
		# Make directory
		first_frame = frames[0]
		has_log = "log" in first_frame
		has_screen = "screen" in first_frame
		if not has_log and not has_screen:
			return
		os.mkdir(episode_directory)
		# Log
		if has_log:
			with open(episode_directory + '/episode.log',"w") as screen_file:
				for i in range(frames_count):
					frame_info = frames[i]
					screen_file.write(frame_info["log"])
		# Screen
		if has_screen:
			screen_filenames = []
			screens_directory = episode_directory+'/screens' 
			os.mkdir(screens_directory)
			for i in range(frames_count):
				filename = screens_directory+'/frame{}'.format(i)
				frame_info_screen = frames[i]["screen"]
				file_list = []
				if 'ASCII' in frame_info_screen:
					ascii_filename = filename+'_ASCII.jpg'
					plt.ascii_image(frame_info_screen['ASCII'], ascii_filename)
					file_list.append(ascii_filename)
				if 'RGB' in frame_info_screen:
					rgb_filename = filename+'_RGB.jpg'
					plt.rgb_array_image(frame_info_screen['RGB'], rgb_filename)
					file_list.append(rgb_filename)
				if 'HeatMap' in frame_info_screen:
					hm_filename = filename+'_HM.jpg'
					plt.heatmap(heatmap=frame_info_screen['HeatMap'], figure_file=hm_filename)
					file_list.append(hm_filename)
				# save file
				file_list_len = len(file_list)
				if file_list_len > 1:
					combined_filename = filename+'.jpg'
					plt.combine_images(images_list=file_list, file_name=combined_filename)
					screen_filenames.append(combined_filename)
				elif file_list_len > 0:
					screen_filenames.append(file_list[0])
			# Gif
			if flags.save_episode_gif and len(screen_filenames) > 0:
				gif_filename = episode_directory+'/episode.gif'
				plt.make_gif(file_list=screen_filenames, gif_path=gif_filename)
				# Delete screens, to save memory
				if flags.delete_screens_after_making_gif:
					shutil.rmtree(screens_directory)
				# Zip GIF, to save memory
				if flags.compress_gif:
					with zipfile.ZipFile(gif_filename+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
						zip.write(gif_filename)
					# Remove unzipped GIF
					os.remove(gif_filename)

	def get_statistics(self):
		stats = self.__client_statistics.get()
		stats.update(self.environment.get_statistics())
		return stats
	
	def log_episode_statistics(self, global_step):
		# Get episode info
		tot_step = self.__episode_info['tot_step']
		tot_reward = self.__episode_info['tot_reward']
		tot_manipulated_reward = self.__episode_info['tot_manipulated_reward']
		tot_extrinsic_reward, tot_intrinsic_reward = tot_reward
		# Update statistics
		episode_stats = {
			'intrinsic_reward_per_step': tot_intrinsic_reward/tot_step,
			'intrinsic_reward': tot_intrinsic_reward,
			'extrinsic_reward_per_step': tot_extrinsic_reward/tot_step,
			'extrinsic_reward': tot_extrinsic_reward,
			'step': tot_step
		}
		tot_value = self.__episode_info['tot_value']
		avg_value = tot_value/tot_step
		if len(avg_value)>1:
			episode_stats.update({
				'extrinsic_value_per_step': avg_value[0],
				'intrinsic_value_per_step': avg_value[1],
			})
		else:
			episode_stats.update({'value_per_step': avg_value[0]})
		self.__client_statistics.add(episode_stats)
		self.stats = self.get_statistics()
		# Print statistics
		self.__reward_logger.info( str(["{0}={1}".format(key,value) for key,value in episode_stats.items()]) )
		# Print frames
		if self.save_frame_info:
			tot_reward = np.around(tot_reward, decimals=1)
			tot_manipulated_reward = np.around(tot_manipulated_reward, decimals=1)
			frames = [self.get_frame_info(step_info) for step_info in self.__episode_step]
			episode_directory = "{}/episodes/reward({}-{})_value_({})_step({})_thread({})".format(flags.log_dir, tot_reward, tot_manipulated_reward, avg_value, global_step, self.environment_id)
			self.print_frames(frames, episode_directory)

	def get_frame_info(self, frame):
		actor = frame['policy']
		distribution = [np.around(softmax(head), decimals=3) for head in actor]
		logits = [np.around(head, decimals=3) for head in actor]
		value = np.around(frame['value'], decimals=3)
		value_info = "reward={}, manipulated_reward={}, value={}\n".format(frame['reward'], frame['manipulated_reward'], value)
		actor_info = "logits={}, distribution={}\n".format(logits, distribution)
		action_info = "action={}\n".format(frame['action'])
		extra_info = "extra={}\n".format(frame['extra'])
		frame_info = { "log": value_info + actor_info + action_info + extra_info }
		if flags.save_episode_screen and frame['screen'] is not None:
			frame_info["screen"] = frame['screen']
		return frame_info
	
	def initialize_new_batch(self):
		self._batch = ExperienceBatch(self.model_size)
		
	def get_internal_states(self):
		return self._last_internal_state
	
	def update_batch_state(self, terminal, internal_state):
		self.terminal = terminal
		self._last_internal_state = internal_state
		
	def apply_action_to_batch(self, agent, action_dict, extrinsic_reward):
		# Build total reward (intrinsic reward is computed later, more efficiently)
		reward = np.array([extrinsic_reward, 0.])
		manipulated_reward = np.array([self.extrinsic_reward_manipulator(extrinsic_reward), 0.])
		# Add action to _batch
		action_dict.update({
			'rewards': reward,
			'manipulated_rewards': manipulated_reward,
		})
		self._batch.add_action(agent_id=agent, feed_dict=action_dict)
		# Save frame info
		if self.save_frame_info:
			self.__episode_step.append({
				'screen': self.environment.get_screen(),
				'extra': self.environment.get_info(),
				'action': action_dict['actions'],
				'policy': action_dict['policies'],
				'value': action_dict['values'],
				'reward': reward,
				'manipulated_reward': manipulated_reward,
			})
			
	def finalize_batch(self, global_step):
		# Terminate _batch
		self._batch.terminal = self.terminal
		# Add batch to episode list
		self._composite_batch.add(self._batch)
		return self._composite_batch
	
	def log_batch(self, global_step, agents):
		# Save _batch info for building statistics
		rewards, values, manipulated_rewards = self._batch.get_all_actions(actions=['rewards','values','manipulated_rewards'], agents=agents)
		self.__episode_info['tot_reward'] += sum(rewards)
		self.__episode_info['tot_manipulated_reward'] += sum(manipulated_rewards)
		self.__episode_info['tot_value'] += sum(values)
		self.__episode_info['tot_step'] += len(rewards)
		# Terminate episode, if _batch is terminal
		if self.terminal: # an episode has terminated
			self.log_episode_statistics(global_step)
