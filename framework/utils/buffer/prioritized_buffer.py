# -*- coding: utf-8 -*-
from random import choice, randint, random
import numpy as np
from sortedcontainers import SortedDict
from utils.buffer.buffer import Buffer

class PrioritizedBuffer(Buffer):
	__slots__ = ('alpha','prefixsum','priorities')
	
	def __init__(self, size, alpha=1):
		self.alpha = alpha # how much prioritization is used (0 - no prioritization, 1 - full prioritization)
		super().__init__(size)
		
	def set(self, buffer):
		assert isinstance(buffer, PrioritizedBuffer)
		super().set(buffer)
	
	def clean(self):
		super().clean()
		self.prefixsum = []
		self.priorities = []
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for batch in self.batches:
				result += list(batch.values())
			return result
		return self.batches[self.get_type(type_id)].values()
		
	def _add_type_if_not_exist(self, type_id):
		if type_id in self.types:
			return False
		self.types[type_id] = len(self.types)
		self.batches.append(SortedDict())
		self.prefixsum.append([])
		self.priorities.append({})
		return True
		
	def get_priority_from_unique(self, unique):
		return float(unique.split('#', 1)[0])
		
	def build_unique(self, priority, count):
		return '{:.5f}#{}'.format(priority,count) # new batch has higher unique priority than old ones with same shared priority
		
	def put(self, batch, priority, type_id=0): # O(log)
		priority_sign = -1 if priority < 0 else 1
		priority = priority_sign*np.power(np.absolute(priority),self.alpha)
		self._add_type_if_not_exist(type_id)
		type = self.get_type(type_id)
		if self.is_full(type):
			index = randint(len(self.batches[type])) if randint(2) == 1 else 0 # argument with lowest priority is always 0 because buffer is sorted by priority
			old_unique_batch_priority, _ = self.batches[type].popitem(index)
			old_priority = self.get_priority_from_unique(old_unique_batch_priority)
			if old_priority in self.priorities[type] and self.priorities[type][old_priority] == 1: # remove from priority dictionary in order to prevent buffer overflow
				del self.priorities[type][old_priority]
		priority_count = self.priorities[type][priority] if priority in self.priorities[type] else 0
		priority_count = (priority_count % self.size) + 1 # modular counter to avoid overflow
		self.priorities[type][priority] = priority_count
		unique_batch_priority = self.build_unique(priority,priority_count)
		self.batches[type].update({unique_batch_priority: batch}) # O(log)
		self.prefixsum[type] = None # compute prefixsum only if needed, when sampling
		
	def keyed_sample(self): # O(n) after a new put, O(log) otherwise
		type_id = choice(self.type_keys)
		type = self.get_type(type_id)
		if self.prefixsum[type] is None: # compute prefixsum
			self.prefixsum[type] = np.cumsum([self.get_priority_from_unique(k) for k in self.batches[type].keys()]) # O(n)
		mass = random() * self.prefixsum[type][-1]
		idx = np.searchsorted(self.prefixsum[type], mass) # O(log) # Find arg of leftmost item greater than or equal to x
		keys = self.batches[type].keys()
		if idx == len(keys): # this may happen when self.prefixsum[type] is negative
			idx = -1
		return self.batches[type][keys[idx]], idx, type_id
		
	def sample(self): # O(n) after a new put, O(log) otherwise
		return self.keyed_sample()[0]

	def update_priority(self, idx, priority, type_id=0): # O(log)
		type = self.get_type(type_id)
		_, batch = self.batches[type].popitem(index=idx) # argument with lowest priority is always 0 because buffer is sorted by priority
		self.put(batch, priority, type_id)
