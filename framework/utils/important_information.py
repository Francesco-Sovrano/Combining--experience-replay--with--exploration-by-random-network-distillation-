# -*- coding: utf-8 -*-
import traceback

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
				if '__dict__' in dir(source_obj):
					for key,value in source_obj.__dict__.items():
						target_obj.__dict__[key] = value
				elif '__slots__' in dir(source_obj):
					for key in source_obj.__slots__:
						value = getattr(source_obj, key)
						setattr(target_obj, key, value)
			except:
				traceback.print_exc()