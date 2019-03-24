# -*- coding: utf-8 -*-
from collections import deque
import gym
from environment.environment import Environment
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False) # prevent opencv to use GPU
import options
flags = options.get()
		
class GymEnvironment(Environment):
	frames_per_state = 4
	max_step = 3000
	state_scaler = 255.
	
	def get_action_shape(self):
		return [(1,len(self.actions_set))] # take 1 action of n possible types
		
	def get_state_shape(self):
		if self.use_ram:
			return [(128,1,self.frames_per_state)]
		return [(42,42,self.frames_per_state)]
	
	def __init__(self, id, environment_name):
		self.id = id
		# setup environment
		self.__game = gym.make(environment_name)
		# collect minimal action set
		if "Montezuma" in environment_name:
			self.actions_set = [1,2,3,4,5,11,12]
		else:
			self.actions_set = list(range(self.__game.action_space.n))
		#=======================================================================
		# For atari games:
		# ACTION_MEANING = {
		#     0 : "NOOP",
		#     1 : "FIRE",
		#     2 : "UP",
		#     3 : "RIGHT",
		#     4 : "LEFT",
		#     5 : "DOWN",
		#     6 : "UPRIGHT",
		#     7 : "UPLEFT",
		#     8 : "DOWNRIGHT",
		#     9 : "DOWNLEFT",
		#     10 : "UPFIRE",
		#     11 : "RIGHTFIRE",
		#     12 : "LEFTFIRE",
		#     13 : "DOWNFIRE",
		#     14 : "UPRIGHTFIRE",
		#     15 : "UPLEFTFIRE",
		#     16 : "DOWNRIGHTFIRE",
		#     17 : "DOWNLEFTFIRE",
		# }
		#=======================================================================
		# evaluator stuff
		self.use_ram = "-ram" in environment_name
		# observation stack
		self.__observation_stack = deque(maxlen=self.frames_per_state)
		self.__state_shape = self.get_state_shape()[0]
		self.__frame_shape = self.__state_shape[:-1]

	def reset(self, data_id=None):
		self.stop()
		observation = self.__game.reset()
		observation = self.normalize(observation)
		self.__observation_stack.clear()
		for _ in range(self.frames_per_state):
			self.__observation_stack.append(observation)
		self.last_state = self.get_state_from_observation_stack()
		self.last_action = None
		self.last_reward = 0
		#=======================================================================
		# self.last_lives = -1
		#=======================================================================
		self.step = 0
		return self.last_state
		
	def normalize(self, observation):
		if not self.use_ram:
			# RGB to Gray
			observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
			# Resize image
			observation = cv2.resize(src=observation, dsize=self.__frame_shape, interpolation=cv2.INTER_AREA)
		else:
			observation = np.reshape(observation, self.__frame_shape)
		return observation
	
	def get_state_from_observation_stack(self):
		return [np.reshape(self.__observation_stack, self.__state_shape)]
		
	def stop(self):
		self.__game.close()
		
	def get_screen(self):
		return {'RGB': self.__game.render('rgb_array')}
		
	def process(self, action_vector):
		observation, reward, is_terminal, info = self.__game.step(self.actions_set[action_vector[0]])
		# build state from observation
		observation = self.normalize(observation)
		self.__observation_stack.append(observation)
		state = self.get_state_from_observation_stack()
		# store last results
		self.last_state = state
		self.last_action = action_vector
		self.last_reward = reward
		# complete step
		self.step += 1
		#=======================================================================
		# lives = info["ale.lives"]
		# if lives < self.last_lives:
		# 	is_terminal = True
		# self.last_lives = lives
		#=======================================================================
		# Check steps constraints, cannot exceed given limit
		if self.step > self.max_step:
			is_terminal = True
		return state, reward, is_terminal, None
	