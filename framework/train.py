# -*- coding: utf-8 -*-
import tensorflow as tf
from agent.server import train

def main(argv):
	train()
	
if __name__ == '__main__':
	tf.app.run()