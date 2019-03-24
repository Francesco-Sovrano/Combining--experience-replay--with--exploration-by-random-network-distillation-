# -*- coding: utf-8 -*-
from random import choice
from collections import deque

class Buffer(object):
	__slots__ = ('size','types','batches','type_values','type_keys')
	
	def __init__(self, size):
		self.size = size
		self.clean()
		
	def clean(self):
		self.types = {}
		self.type_values = []
		self.type_keys = [] 
		self.batches = []
		
	def _add_type_if_not_exist(self, type_id): # private method
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = len(self.types)
		self.type_values.append(self.types[type_id])
		self.type_keys.append(type_id)
		self.batches.append(deque(maxlen=self.size))
		return True
		
	def set(self, buffer):
		assert isinstance(buffer, Buffer)
		for key in self.__slots__:
			setattr(self, key, getattr(buffer, key))
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for batch in self.batches:
				result += batch
			return result
		return self.batches[self.get_type(type_id)]

	def has_atleast(self, frames, type=None):
		return self.count(type) >= frames
		
	def has(self, frames, type=None):
		return self.count(type) == frames
		
	def count(self, type=None):
		if type is None:
			if len(self.batches) == 0:
				return 0
			return sum(len(batch) for batch in self.batches)
		return len(self.batches[type])
		
	def id_is_full(self, type_id):
		return self.has(self.size, self.get_type(type_id))
		
	def is_full(self, type=None):
		if type is None:
			return self.has(self.size*len(self.types))
		return self.has(self.size, type)
		
	def is_empty(self, type=None):
		return not self.has_atleast(1, type)
		
	def get_type(self, type_id):
		return self.types[type_id]

	def put(self, batch, type_id=0): # put batch into buffer
		self._add_type_if_not_exist(type_id)
		type = self.get_type(type_id)
		self.batches[type].append(batch)

	def sample(self):
		type = choice(self.type_values)
		return choice(self.batches[type])
