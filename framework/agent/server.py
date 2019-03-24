# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.tensorflow_utils as tf_utils
import numpy as np
from threading import Thread, RLock, Timer
import os
import traceback
import logging
import time
import sys
from pickle import Pickler, Unpickler
import gc
from multiprocessing import Queue, Process
from collections import deque
from agent.worker.working_group import Group
import utils.plots as plt
from utils.misc import get_cpu_count
import psutil
from copy import deepcopy
from utils.important_information import ImportantInformation
from utils.statistics import Statistics, IndexedStatistics
from collections import Counter
import linecache
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from time import sleep
from environment.environment import Environment
#===============================================================================
# import signal
#===============================================================================
import options
flags = options.get()

def train():
	result_queue = Queue()
	p = Process(target=lambda q: q.put(Application().train()), args=(result_queue,))
	p.start()
	p.join()
	if not result_queue.empty():
		is_alive = result_queue.get()
		if is_alive:
			train()
			
def display_top(snapshot, key_type='lineno', limit=3):
	snapshot = snapshot.filter_traces((
		tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
		tracemalloc.Filter(False, "<unknown>"),
	))
	top_stats = snapshot.statistics(key_type)

	print("Top %s lines" % limit)
	for index, stat in enumerate(top_stats[:limit], 1):
		frame = stat.traceback[0]
		# replace "/path/to/module/file.py" with "module/file.py"
		filename = os.sep.join(frame.filename.split(os.sep)[-2:])
		print("#%s: %s:%s: %.1f KiB"
			  % (index, filename, frame.lineno, stat.size / 1024))
		line = linecache.getline(frame.filename, frame.lineno).strip()
		if line:
			print('	%s' % line)

	other = top_stats[limit:]
	if other:
		size = sum(stat.size for stat in other)
		print("%s other: %.1f KiB" % (len(other), size / 1024))
	total = sum(stat.size for stat in top_stats)
	print("Total allocated size: %.1f KiB" % (total / 1024))

def memory_monitor():
	tracemalloc.start()
	old_max = 0
	while True:
		sleep(60)
		try:
			max_rss = getrusage(RUSAGE_SELF).ru_maxrss
			if max_rss > old_max:
				old_max = max_rss
				snapshot = tracemalloc.take_snapshot()
				print(datetime.now(), 'max RSS', max_rss)
				display_top(snapshot)
		except:
			traceback.print_exc()			

class Application(object):
					
	def __init__(self):
		if not os.path.isdir(flags.log_dir):
			os.mkdir(flags.log_dir)
		self.train_logfile = flags.log_dir + '/train_results.log'
		# Training logger
		self.training_logger = logging.getLogger('results')
		hdlr = logging.FileHandler(self.train_logfile)
		hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
		self.training_logger.addHandler(hdlr) 
		self.training_logger.setLevel(logging.DEBUG)
		# Initialize network
		self.num_cpu = get_cpu_count()
		self.num_gpu = tf_utils.gpu_count()
		print("Available CPUs: {}".format(self.num_cpu))
		print("Available GPUs: {}".format(self.num_gpu))
		self.thread_count = min(flags.environment_count,flags.threads_per_cpu*self.num_cpu) if flags.threads_per_cpu > 0 else 1
		self.env_per_worker = max(1,flags.environment_count//self.thread_count) 
		self.is_alive = True
		self.terminate_requested = False
		self.process = psutil.Process(os.getpid())
		self.checkpoint_list = deque()
		self.performance_timer = None
		# Set start time
		self.start_time = time.time()
		# Build network
		self.build_network()
		if flags.monitor_memory_usage:
			monitor_thread = Thread(target=memory_monitor)
			monitor_thread.start()
		
	def create_session(self):
		print('Creating new tensorflow session..')
		# GPU options
		gpu_options = tf.GPUOptions(
			allow_growth = True,
			# per_process_gpu_memory_fraction=1/self.num_cpu,
		)
		# Config proto
		config_proto = tf.ConfigProto(
			# https://www.tensorflow.org/guide/performance/overview
			intra_op_parallelism_threads = self.num_cpu,
			inter_op_parallelism_threads = self.num_cpu,
			allow_soft_placement=True,
			gpu_options=gpu_options,
			#===================================================================
			# log_device_placement=True,
			#===================================================================
		)
		return tf.InteractiveSession(config=config_proto)
		
	def build_global_network(self):
		global_network = Group(group_id=0, environment_count=0, global_network=None, training=True).network_manager
		variables_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		return global_network, variables_to_save

	def build_network(self):
		# Create session
		tf_session = self.create_session()
		# Build global network
		self.global_network, variables_to_save = self.build_global_network()
		# Build local networks
		self.working_groups = [
			Group(group_id=c+1, environment_count=self.env_per_worker, global_network=self.global_network, training=True) 
			for c in range(self.thread_count)
		]
		# Statistics
		self.training_statistics = IndexedStatistics(max_count=self.thread_count, buffer_must_be_full=True)
		# Load checkpoint
		self.saver = tf.train.Saver(
			var_list=variables_to_save, max_to_keep=flags.max_checkpoint_to_keep,
			restore_sequentially=True # Causes restore of different variables to happen sequentially within each device. This can lower memory usage when restoring very large models.
		)
		self.global_step, self.start_elapsed_time = self.load_checkpoint(self.saver, tf_session)
		self.last_global_step = self.global_step
		self.next_save_steps = self.global_step + flags.save_interval_step
		# Print graph summary
		tf.summary.FileWriter('summary', tf_session.graph).close()
		
	def train(self, initialize=True):
		# Initialize threads
		if initialize:
			self.step_lock = RLock() # The standard Lock doesn’t know which thread is currently holding the lock. If the lock is held, any thread that attempts to acquire it will block, even if the same thread itself is already holding the lock.
		tf_session = tf.get_default_session()
		self.train_threads = [Thread(target=self.train_function, args=(i,tf_session)) for i in range(self.thread_count)]
		# Run training threads
		for t in self.train_threads:
			t.start()
		# Init
		if initialize:
			#===================================================================
			# # Set signal handler
			# signal.signal(signal.SIGINT, self.signal_handler)
			# print('Press Ctrl+C to stop')
			#===================================================================
			self.print_performance()		
		# Wait for all threads to stop
		for t in self.train_threads:
			t.join()
		# Save checkpoint
		try:
			self.save_checkpoint(self.global_step, self.saver, tf_session)
			# Test
			if flags.test_after_saving:
				self.test()
		except:
			traceback.print_exc()
		# Restart workers
		if self.is_alive and not flags.rebuild_network_after_checkpoint_is_saved:
			self.next_save_steps += flags.save_interval_step
			return self.train(initialize=False)
		if self.performance_timer is not None:
			self.performance_timer.cancel()
		return self.is_alive

	def train_function(self, parallel_index, session):
		""" Train each environment. """
		group = self.working_groups[parallel_index]
		
		if self.global_step == 0:
			group.initialize_environments(step_count=flags.timesteps_before_starting_training)
	
		while True:
			# Work
			try:
				# Setup default session
				with session.as_default():
					# Process group
					thread_steps = group.process(global_step=self.global_step, batch=True)
				# Update shared memory, lock
				with self.step_lock:
					self.global_step += thread_steps
					# Print global statistics, after training if needed # Ignore the initial logs, because they are too noisy
					if group.has_terminal_worker and group.terminated_episodes > flags.episode_count_for_evaluation:
						self.training_statistics.set(group.get_statistics(), parallel_index)
						info = self.training_statistics.get()
						if info:
							# Print statistics
							self.training_logger.info("<{}> {}".format(self.global_step, ["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])]))
			except:
				traceback.print_exc()
			# Check whether training is still alive
			if self.global_step >= flags.max_timestep or self.terminate_requested:
				self.is_alive = False
				for group in trainer_list:
					group.stop()
				return
			# Save checkpoint
			if self.global_step >= self.next_save_steps:
				return

	def test(self):
		result_file = '{}/test_results_{}.log'.format(flags.log_dir,self.global_step)
		if os.path.exists(result_file):
			print('Test results already produced and evaluated for {}'.format(result_file))
			return
		result_lock = RLock()
			
		print('Start testing')
		testers = []
		threads = []
		tf_session = tf.get_default_session()
		tmp_environment = Environment.create_environment(env_type=flags.env_type, training=False)
		dataset_size = tmp_environment.get_dataset_size()
		data_per_thread = max(1, dataset_size//self.thread_count)
		for i in range(self.thread_count): # parallel testing
			tester = Group(group_id=-(i+1), environment_count=data_per_thread, global_network=self.global_network, training=False)
			data_range_start = i*data_per_thread
			data_range_end = data_range_start + data_per_thread
			# print(data_range_start, data_per_thread, dataset_size)
			thread = Thread(
				target=self.test_function, 
				args=(
					result_file, result_lock,
					tester,
					(data_range_start, data_range_end),
					tf_session
				)
			)
			thread.start()
			threads.append(thread)
			testers.append(tester)
		print('Test Set size:', dataset_size)
		print('Tests per thread:', data_per_thread)
		time.sleep(5)
		for thread in threads: # wait for all threads to end
			thread.join()
		print('End testing')
		# get overall statistics
		test_statistics = Statistics(self.thread_count)
		for group in testers:
			test_statistics.add(group.get_statistics())
		info = test_statistics.get()
		# write results to file
		stats_file = '{}/test_statistics.log'.format(flags.log_dir)
		with open(stats_file, "a", encoding="utf-8") as file: # write stats to file
			file.write('{}\n'.format(["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])]))
		print('Test statistics saved in {}'.format(stats_file))
		print('Test results saved in {}'.format(result_file))
		return tmp_environment.evaluate_test_results(result_file)
	
	def test_function(self, result_file, result_lock, tester, data_range, session):
		with session.as_default():
			for data_id in range(*data_range, tester.environment_count):
				tester.process(batch=False, data_id=data_id)
				if flags.print_test_results:
					result_list = (str(worker.environment.get_test_result()) for worker in tester.worker_list)
					result_string = '\n'.join(result_list)
					with result_lock:
						with open(result_file, "a", encoding="utf-8") as file: # write results to file
							file.write('{}\n'.format(result_string))
				
	def print_performance(self):
		step_delta = self.global_step - self.last_global_step
		self.last_global_step = self.global_step
		ram_usage_in_gb = self.process.memory_info().rss/2**30
		elapsed_time = self.get_elapsed_time()
		steps_per_sec_tot = self.global_step / elapsed_time
		steps_per_sec_now = step_delta / flags.seconds_to_wait_for_printing_performance
		print("### [Used {:.2f} GB] {} STEPS in {:.0f} sec. with {:.0f} STEPS/sec. and {:.0f} dSTEPS/sec.".format(ram_usage_in_gb, self.global_step, elapsed_time, steps_per_sec_tot, steps_per_sec_now))
		if len(gc.garbage) > 0:
			print('Cannot collect {} garbage objects'.format(len(gc.garbage)))
		sys.stdout.flush() # force print immediately what is in output buffer
		if self.is_alive:
			self.performance_timer = Timer(flags.seconds_to_wait_for_printing_performance, self.print_performance)
			self.performance_timer.start()
	
	def load_checkpoint(self, saver, tf_session):
		# Initialize network variables
		tf_session.run(tf.global_variables_initializer()) # do it before loading checkpoint
		# Initialize or load checkpoint with saver
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			print("Loading checkpoint", checkpoint.model_checkpoint_path)
			saver.restore(tf_session, checkpoint.model_checkpoint_path)
			tokens = checkpoint.model_checkpoint_path.split("-")
			# set global step
			global_step = int(tokens[1])
			print(">>> global step set: ", global_step)
			# set wall time
			elapsed_time_fname = '{}/elapsed_time.{}'.format(flags.checkpoint_dir, global_step)
			with open(elapsed_time_fname, 'r') as f:
				start_elapsed_time = float(f.read())
			self.load_important_information(flags.checkpoint_dir + '/{}.pkl'.format(global_step))
			print("Checkpoint loaded")
		else:	
			print("Could not find old checkpoint")
			global_step, start_elapsed_time = 0, 0
		# Finalize graph
		#=======================================================================
		# tf_session.graph.finalize()
		#=======================================================================
		return global_step, start_elapsed_time
			
	def save_checkpoint(self, global_step, saver, tf_session):
		# Create checkpoint directory
		if not os.path.exists(flags.checkpoint_dir):
			os.mkdir(flags.checkpoint_dir)
		# Delete old checkpoints
		self.checkpoint_list.append(global_step)
		if len(self.checkpoint_list) > flags.max_checkpoint_to_keep:
			checkpoint_to_delete = self.checkpoint_list.popleft()
			# Delete the old pickle files, the other checkpoint files are automatically deleted by tensorflow
			os.remove('{}/{}.pkl'.format(flags.checkpoint_dir,checkpoint_to_delete))
			# Delete the old wall time files
			os.remove('{}/elapsed_time.{}'.format(flags.checkpoint_dir,checkpoint_to_delete))
			# The other checkpoint files are automatically deleted by tensorflow
		# Write wall time
		elapsed_time_fname = '{}/elapsed_time.{}'.format(flags.checkpoint_dir, global_step)
		with open(elapsed_time_fname, 'w') as f:
			f.write(str(self.get_elapsed_time()))
		# Print plot
		if flags.compute_plot_when_saving:
			plt.plot_files(log_files=[self.train_logfile], figure_file=flags.log_dir + '/train_plot.jpg')
		# Save Checkpoint
		print('Start saving..')
		saver.save(
			sess=tf_session, 
			save_path='{}/checkpoint'.format(flags.checkpoint_dir), 
			global_step=global_step
		)
		self.save_important_information('{}/{}.pkl'.format(flags.checkpoint_dir, global_step))
		print('Checkpoint saved in {}'.format(flags.checkpoint_dir))
			
	def get_elapsed_time(self):
		return time.time() - self.start_time + self.start_elapsed_time
	
	def save_important_information(self, path):
		# Write
		with open(path, 'wb') as file:
			Pickler(file, protocol=-1).dump(deepcopy(ImportantInformation.get()))
		# Collect garbage
		gc.collect()
			
	def load_important_information(self, path):
		# Read pickle file 
		with open(path, 'rb') as file:
			ImportantInformation.set(deepcopy(Unpickler(file).load()))
		# Collect garbage
		gc.collect()
		
	#===========================================================================
	# def signal_handler(self, signal, frame):
	# 	print('You pressed Ctrl+C!')
	# 	self.terminate_requested = True
	#===========================================================================