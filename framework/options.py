# -*- coding: utf-8 -*-
import tensorflow as tf

options_built = False
def build():
	tf.app.flags.DEFINE_integer("max_timestep", 2**30, "Max training time steps")
	tf.app.flags.DEFINE_integer("timesteps_before_starting_training", 2**10, "Max training time steps")
# Environment
	# tf.app.flags.DEFINE_string("env_type", "car_controller", "environment types: rogue, car_controller, sentipolc, or environments from https://gym.openai.com/envs")
	tf.app.flags.DEFINE_string("env_type", "MontezumaRevengeDeterministic-v4", "Environment types: rogue, car_controller, sentipolc, or environments from https://gym.openai.com/envs")
	# tf.app.flags.DEFINE_string("env_type", "MultipleSequenceAlignment-BaliBase", "Environment types: rogue, car_controller, sentipolc, MultipleSequenceAlignment-BaliBase, or environments from https://gym.openai.com/envs")
	# tf.app.flags.DEFINE_string("env_type", "sentipolc", "environment types: rogue, car_controller, sentipolc, or environments from https://gym.openai.com/envs")
	# tf.app.flags.DEFINE_string("env_type", "rogue", "environment types: rogue, car_controller, sentipolc, or environments from https://gym.openai.com/envs")
# Gradient optimization parameters
	tf.app.flags.DEFINE_string("parameters_type", "float32", "The type used to represent parameters: float32, float64")
	tf.app.flags.DEFINE_string("algorithm", "AC", "algorithms: AC, ACER")
	tf.app.flags.DEFINE_string("network_configuration", "OpenAILarge", "neural network configurations: Base, Towers, HybridTowers, SA, OpenAISmall, OpenAILarge, Impala")
	tf.app.flags.DEFINE_boolean("network_has_internal_state", False, "Whether the network has an internal state to keep updated (eg. RNNs state).")
	tf.app.flags.DEFINE_string("optimizer", "Adam", "gradient optimizer: PowerSign, AddSign, ElasticAverage, LazyAdam, Nadam, Adadelta, AdagradDA, Adagrad, Adam, Ftrl, GradientDescent, Momentum, ProximalAdagrad, ProximalGradientDescent, RMSProp") # default is Adam, for vanilla A3C is RMSProp
	# In information theory, the cross entropy between two probability distributions p and q over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set.
	tf.app.flags.DEFINE_boolean("only_non_negative_entropy", True, "Cross-entropy and entropy are used for policy loss and if this flag is true, then entropy=max(0,entropy). If cross-entropy measures the average number of bits needed to identify an event, then it cannot be negative.")
	# Use mean losses if max_batch_size is too big, in order to avoid NaN
	tf.app.flags.DEFINE_string("loss_type", "mean", "type of loss reduction: sum, mean")
	tf.app.flags.DEFINE_string("policy_loss", "PPO", "policy loss function: Vanilla, PPO")
	tf.app.flags.DEFINE_string("value_loss", "PVO", "value loss function: Vanilla, PVO")
# Loss clip range
	tf.app.flags.DEFINE_float("clip", 0.1, "PPO/PVO initial clip range") # default is 0.2, for openAI is 0.1
	tf.app.flags.DEFINE_boolean("clip_decay", False, "Whether to decay the clip range")
	tf.app.flags.DEFINE_string("clip_annealing_function", "exponential_decay", "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay") # default is inverse_time_decay
	tf.app.flags.DEFINE_integer("clip_decay_steps", 10**5, "decay clip every x steps") # default is 10**6
	tf.app.flags.DEFINE_float("clip_decay_rate", 0.96, "decay rate") # default is 0.25
# Learning rate
	tf.app.flags.DEFINE_float("alpha", 1e-4, "initial learning rate") # default is 7.0e-4, for openAI is 2.5e-4
	tf.app.flags.DEFINE_boolean("alpha_decay", False, "whether to decay the learning rate")
	tf.app.flags.DEFINE_string("alpha_annealing_function", "exponential_decay", "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay") # default is inverse_time_decay
	tf.app.flags.DEFINE_integer("alpha_decay_steps", 10**8, "decay alpha every x steps") # default is 10**6
	tf.app.flags.DEFINE_float("alpha_decay_rate", 0.96, "decay rate") # default is 0.25
# Intrinsic Rewards: Burda, Yuri, et al. "Exploration by Random Network Distillation." arXiv preprint arXiv:1810.12894 (2018).
	tf.app.flags.DEFINE_boolean("intrinsic_reward", True, "An intrinisc reward is given for exploring new states.")
	tf.app.flags.DEFINE_boolean("split_values", True, "Estimate separate values for extrinsic and intrinsic rewards.")
	tf.app.flags.DEFINE_integer("intrinsic_reward_step", 2**20, "Start using the intrinsic reward only when global step is greater than n.")
	tf.app.flags.DEFINE_boolean("scale_intrinsic_reward", False, "Whether to scale the intrinsic reward with its standard deviation.")
	tf.app.flags.DEFINE_float("intrinsic_rewards_mini_batch_fraction", 0, "Keep only the best intrinsic reward in a mini-batch of size 'batch_size*fraction', and set other intrinsic rewards to 0.")
	tf.app.flags.DEFINE_float("intrinsic_reward_gamma", 0.99, "Discount factor for intrinsic rewards") # default is 0.95, for openAI is 0.99
	tf.app.flags.DEFINE_float("extrinsic_coefficient", 2., "Scale factor for the extrinsic part of the advantage.")
	tf.app.flags.DEFINE_float("intrinsic_coefficient", 1., "Scale factor for the intrinsic part of the advantage.")
	tf.app.flags.DEFINE_boolean("episodic_extrinsic_reward", True, "Bootstrap 0 for extrinsic value if state is terminal.")
	tf.app.flags.DEFINE_boolean("episodic_intrinsic_reward", False, "Bootstrap 0 for intrinsic value if state is terminal.")
# Experience Replay
	# Replay mean > 0 increases off-policyness
	tf.app.flags.DEFINE_float("replay_mean", 0.5, "Mean number of experience replays per batch. Lambda parameter of a Poisson distribution. When replay_mean is 0, then experience replay is not active.") # for A3C is 0, for ACER default is 4
	tf.app.flags.DEFINE_integer("replay_step", 2**20, "Start replaying experience when global step is greater than replay_step.")
	tf.app.flags.DEFINE_integer("replay_buffer_size", 2**7, "Maximum number of batches stored in the experience buffer.")
	tf.app.flags.DEFINE_integer("replay_start", 1, "Buffer minimum size before starting replay. Should be greater than 0 and lower than replay_buffer_size.")
	tf.app.flags.DEFINE_boolean("replay_only_best_batches", False, "Whether to replay only those batches leading to an extrinsic reward (the best ones).")
	tf.app.flags.DEFINE_boolean("constraining_replay", False, "Use constraining replay loss for the Actor, in order to minimize the quadratic distance between the sampled batch actions and the Actor mean actions (softmax output). ")
	tf.app.flags.DEFINE_boolean("recompute_value_when_replaying", False, "Whether to recompute values, advantages and discounted cumulative rewards when replaying, even if not required by the model.")
	tf.app.flags.DEFINE_boolean("train_critic_when_replaying", True, "Whether to train also the critic when replaying. Works only when separate_actor_from_critic=True.")
	tf.app.flags.DEFINE_boolean("runtime_advantage", True, "Whether to compute advantage at runtime, using always up to date state values instead of old ones.") # default False
	# tf.app.flags.DEFINE_float("loss_stationarity_range", 5e-3, "Used to decide when to interrupt experience replay. If the mean actor loss is whithin this range, then no replay is performed.")
# Prioritized Experience Replay: Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
	tf.app.flags.DEFINE_boolean("prioritized_replay", True, "Whether to use prioritized sampling (if replay_mean > 0)") # default is True
	tf.app.flags.DEFINE_float("prioritized_replay_alpha", 0.5, "How much prioritization is used (0 - no prioritization, 1 - full prioritization).")
	tf.app.flags.DEFINE_float("prioritized_drop_probability", 1, "Probability of removing the batch with the lowest priority instead of the oldest batch.")
# Reward manipulators
	tf.app.flags.DEFINE_string("extrinsic_reward_manipulator", 'lambda x: np.clip(x,-1,1)', "Set to 'lambda x: x' for no manipulation. A lambda expression used to manipulate the extrinsic rewards.")
	tf.app.flags.DEFINE_string("intrinsic_reward_manipulator", 'lambda x: x', "Set to 'lambda x: x' for no manipulation. A lambda expression used to manipulate the intrinsic rewards.")
# Actor-Critic parameters
	tf.app.flags.DEFINE_boolean("separate_actor_from_critic", False, "Set to True if you want actor and critic not sharing any part of their computational graphs.") # default False
	tf.app.flags.DEFINE_float("value_coefficient", 1, "Value coefficient for tuning Critic learning rate.") # default is 0.5
	tf.app.flags.DEFINE_integer("environment_count", 128, "Number of different parallel environments, used for training.")
	tf.app.flags.DEFINE_integer("threads_per_cpu", 1, "Number of threads per CPU. Set to 0 to use only one CPU.")
	tf.app.flags.DEFINE_integer("batch_size", 128, "Maximum batch size.") # default is 8
	# A big enough big_batch_size can significantly speed up the algorithm when training on GPU
	tf.app.flags.DEFINE_integer("big_batch_size", 16, "Number n > 0 of batches that compose a big-batch used for training. The bigger is n the more is the memory consumption.")
	# Taking gamma < 1 introduces bias into the policy gradient estimate, regardless of the value function accuracy.
	tf.app.flags.DEFINE_float("gamma", 0.999, "Discount factor for extrinsic rewards") # default is 0.95, for openAI is 0.99
# Entropy regularization
	tf.app.flags.DEFINE_boolean("entropy_regularization", True, "Whether to add entropy regularization to policy loss.") # default True
	tf.app.flags.DEFINE_float("beta", 1e-3, "entropy regularization constant") # default is 0.001, for openAI is 0.01
# Generalized Advantage Estimation: Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
	tf.app.flags.DEFINE_boolean("use_GAE", True, "Whether to use Generalized Advantage Estimation.") # default in openAI's PPO implementation
	# Taking lambda < 1 introduces bias only when the value function is inaccurate
	tf.app.flags.DEFINE_float("lambd", 0.95, "generalized advantage estimator decay parameter") # default is 0.95
# Log
	tf.app.flags.DEFINE_integer("save_interval_step", 2**22, "Save a checkpoint every n steps.")
	# rebuild_network_after_checkpoint_is_saved may help saving RAM, but may be slow proportionally to save_interval_step.
	tf.app.flags.DEFINE_boolean("rebuild_network_after_checkpoint_is_saved", False, "Rebuild the whole network after checkpoint is saved. This may help saving RAM, but it's slow.")
	tf.app.flags.DEFINE_integer("max_checkpoint_to_keep", 3, "Keep the last n checkpoints, delete the others")
	tf.app.flags.DEFINE_boolean("test_after_saving", False, "Whether to test after saving")
	tf.app.flags.DEFINE_boolean("print_test_results", False, "Whether to print test results when testing")
	tf.app.flags.DEFINE_integer("episode_count_for_evaluation", 2**5, "Number of matches used for evaluation scores")
	tf.app.flags.DEFINE_integer("seconds_to_wait_for_printing_performance", 60, "Number of seconds to wait for printing algorithm performance in terms of memory and time usage")
	tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint directory")
	tf.app.flags.DEFINE_string("event_dir", "./events", "events directory")
	tf.app.flags.DEFINE_string("log_dir", "./log", "events directory")
	tf.app.flags.DEFINE_boolean("print_loss", True, "Whether to print losses inside statistics") # print_loss = True might slow down the algorithm
	tf.app.flags.DEFINE_boolean("print_policy_info", True, "Whether to print debug information about the actor inside statistics") # print_policy_info = True might slow down the algorithm
	tf.app.flags.DEFINE_string("show_episodes", 'random', "What type of episodes to save: random, best, all, none")
	tf.app.flags.DEFINE_float("show_episode_probability", 2e-3, "Probability of showing an episode when show_episodes == random")
	# save_episode_screen = True might slow down the algorithm -> use in combination with show_episodes = 'random' for best perfomance
	tf.app.flags.DEFINE_boolean("save_episode_screen", True, "Whether to save episode screens")
	# save_episode_gif = True slows down the algorithm, requires save_episode_screen = True to work
	tf.app.flags.DEFINE_boolean("save_episode_gif", True, "Whether to save episode GIF, requires save_episode_screen == True.")
	tf.app.flags.DEFINE_float("gif_speed", 0.1, "GIF frame speed in seconds.")
	tf.app.flags.DEFINE_boolean("compress_gif", False, "Whether to zip the episode GIF.")
	tf.app.flags.DEFINE_boolean("delete_screens_after_making_gif", True, "Whether to delete the screens after the GIF has been made.")
	tf.app.flags.DEFINE_boolean("monitor_memory_usage", False, "Whether to monitor memory usage")
# Plot
	tf.app.flags.DEFINE_boolean("compute_plot_when_saving", True, "Whether to compute the plot when saving checkpoints")
	tf.app.flags.DEFINE_integer("max_plot_size", 25, "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead.")
	
	global options_built
	options_built = True
	
def get():
	global options_built
	if not options_built:
		build()
	return tf.app.flags.FLAGS