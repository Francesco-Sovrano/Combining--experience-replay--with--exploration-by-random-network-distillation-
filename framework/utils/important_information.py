# -*- coding: utf-8 -*-
import traceback
from itertools import chain

class ImportantInformation(object):
	instances = {}
	
	def __init__(self, obj, name):
		assert isinstance(obj, object), 'Important information must be an object'
		self.instances[name] = obj
		
	@staticmethod
	def get():
		return ImportantInformation.instances
		
	@staticmethod
	def set(instances):
		for name, source_obj in instances.items():
			print('Loading important object:',name)
			try:
				target_obj = ImportantInformation.instances[name]
				ImportantInformation.load(source_obj,target_obj)
			except:
				traceback.print_exc()
				
	@staticmethod
	def load(source_obj,target_obj):
		vtype = type(source_obj)
		# loading lists and tuples
		if vtype in [list,tuple]:
			for s,t in zip(source_obj,target_obj):
				ImportantInformation.load(s,t)
		# loading dictionaries
		elif vtype is dict:
			for key,value in source_obj.items():
				target_obj[key] = value
		# loading other objects
		else:		
			dicts = list(source_obj.__dict__.keys()) if hasattr(source_obj, '__dict__') else []
			slots = list(chain.from_iterable(getattr(cls, '__slots__', []) for cls in source_obj.__class__.__mro__))
			print('	..slots',slots)
			print('	..dicts',dicts)
			# get slot from current class and parent classes (if any)
			for key in slots+dicts:
				# print('	..loading attribute', key)
				value = getattr(source_obj, key)
				setattr(target_obj, key, value)