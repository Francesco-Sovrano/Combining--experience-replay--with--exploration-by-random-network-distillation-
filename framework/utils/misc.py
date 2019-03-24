# -*- coding: utf-8 -*-
import operator
import os
from multiprocessing import cpu_count
import io
import numpy as np
from threading import RLock

def accumulate(iterable, func=operator.add, initial_value=0.):
	total = initial_value
	it = iter(iterable)
	try:
		for element in it:
			total = func(total, element)
			yield total
	except StopIteration:
		return
	
def get_padded_size(size, kernel, stride):
	return kernel if size <= kernel else size + (stride - (size - kernel)%stride)%stride 
	
def flatten(l): 
	return [item for sublist in l for item in sublist]

def softmax(logits, axis=-1):
	val = np.exp(logits)
	return val/np.sum(val, axis=axis, keepdims=True)
	
def is_tuple(val):
	return type(val) in [list,tuple]

def compress(val):
	vtype = type(val)
	if vtype in [list,tuple]:
		return vtype(compress(v) for v in val)
	elif vtype is np.ndarray:
		compressed_array = io.BytesIO()
		np.savez_compressed(compressed_array, val)
		return compressed_array
	return val

def decompress(val):
	vtype = type(val)
	if vtype in [list,tuple]:
		return vtype(decompress(v) for v in val)
	elif vtype is io.BytesIO:
		val.seek(0) # seek back to the beginning of the file-like object
		return np.load(val)['arr_0']
	return val
		
def get_cpu_count():
	return int(os.getenv('RCALL_NUM_CPU', cpu_count()))

class BoolLock(object): # Easily activate/deactivate locking without changes to the code
	def __init__(self, use_lock=False):
		self.use_lock = use_lock
		self.lock = RLock()
		
	def release(self):
		if self.use_lock:
			self.lock.release()
			
	def acquire(self, blocking=True, timeout=-1):
		if self.use_lock:
			self.lock.acquire(blocking, timeout)

	def __enter__(self, blocking=True, timeout=-1):
		return self if not self.use_lock else self.lock.__enter__(blocking, timeout)

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.use_lock:
			self.lock.__exit__(exc_type, exc_val, exc_tb)
