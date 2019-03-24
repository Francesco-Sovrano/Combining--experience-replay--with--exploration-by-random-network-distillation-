# -*- coding: utf-8 -*-
from collections import deque
from sklearn.preprocessing import StandardScaler
import numpy as np

class RunningMeanStd(object):
	
	def __init__(self, batch_size=0, shape=()):
		self.shape = shape
		self.do_reshape = len(self.shape) > 1
		self.scaler = StandardScaler(copy=False)
		self.batch_size = batch_size
		if self.batch_size > 0:
			self.batch = deque(maxlen=self.batch_size)
		self.initialized = False
		self.mean = np.zeros(self.shape, np.float16)
		self.var = np.ones(self.shape, np.float16)
		self.std = np.ones(self.shape, np.float16)
		
	def copy(self):
		new_obj = RunningMeanStd(batch_size=self.batch_size)
		new_obj.scaler = self.scaler
		new_obj.initialized = self.initialized
		# return without copying the batch, for saving memory
		return new_obj

	def update(self, x):
		if self.batch_size > 0:
			self.batch.extend(x)
			if len(self.batch) >= self.batch_size:
				batch = self.batch
				self.batch = deque(maxlen=self.batch_size)
			else:
				batch = None
		else:
			batch = x
		if batch is None:
			return False
		batch = np.reshape(batch, (len(batch),-1)).astype(np.float64)
		self.scaler.partial_fit(batch)
		self.mean = self.scaler.mean_.astype(np.float16)
		self.var = self.scaler.var_.astype(np.float16)
		self.std = self.scaler.scale_.astype(np.float16)
		if self.do_reshape:
			self.mean = np.reshape(self.mean, self.shape) 
			self.var = np.reshape(self.var, self.shape)
			self.std = np.reshape(self.std, self.shape)
		self.initialized = True
		return True
