# -*- coding: utf-8 -*-
from collections import deque
import numpy as np

class Statistics(object):
	
	def __init__(self, max_count, buffer_must_be_full=False):
		self._max_count = max_count
		self._stats = {}
		self._buffer_must_be_full = buffer_must_be_full
		
	def add(self, stat_dict, type=''):
		if type not in self._stats:
			self._stats[type] = deque(maxlen=self._max_count)
		self._stats[type].append(stat_dict)
		
	def buffer_is_full(self, type):
		return len(self._stats[type]) == self._max_count
	
	def get(self):
		final_dict = {}
		for type,stat in self._stats.items():
			if not stat:
				continue
			if self._buffer_must_be_full and not self.buffer_is_full(type):
				continue
			biggest_dictionary = max(stat, key=lambda x: len(x.keys()))
			keys = biggest_dictionary.keys()
			final_dict.update({'{1}{0}'.format(k,type): np.mean([e[k] for e in stat if k in e]) for k in keys})
		return final_dict
	
class IndexedStatistics(Statistics):
	
	def __init__(self, max_count, buffer_must_be_full=False):
		super().__init__(max_count, buffer_must_be_full)
		self._non_empty_stats = {}
	
	def add(self, stat_dict, type=''):
		raise AttributeError
	
	def set(self, stat_dict, index, type=''):
		if type not in self._stats:
			self._stats[type] = [None]*self._max_count
			self._non_empty_stats[type] = 0
		stat_was_empty = self._stats[type][index] is None
		self._stats[type][index] = stat_dict
		stat_is_empty = stat_dict is None
		if stat_was_empty:
			if not stat_is_empty:
				self._non_empty_stats[type] += 1
		elif stat_is_empty:
			self._non_empty_stats[type] -= 1
	
	def buffer_is_full(self, type):
		return self._non_empty_stats[type] == self._max_count