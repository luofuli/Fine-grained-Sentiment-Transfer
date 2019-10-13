# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2018/8/6 20:56 
"""
import argparse
from common_options import *


def add_s2ss_arguments(parser):
    parser.add_argument("--mode", default="train", help="train, inference, final_inference")
    parser.add_argument("--n_epoch", default=3, type=int, help="Max n epoch during training.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size of training.")
    parser.add_argument("--cell_type", default='lstm', help='Cell type must in [lstm, gru]')
    parser.add_argument("--num_hidden", default=256, help="Number of hidden size")
    parser.add_argument("--decode_type", default="greedy", help="Type of decode: `greedy`, `random`.")
    parser.add_argument("--decode_width", default=5, type=int, help="Width of the beam search or random search.")
    parser.add_argument('--MLE_learning_rate', default=0.0001, type=float, help='Learning rate of MLE')
    parser.add_argument('--RL_learning_rate', default=0.000001, type=float, help='Learning rate of RL')
    parser.add_argument("--softmax_temperature", default=0.001, type=float,
                        help='Softmax temperature in Gaussian layer')
    parser.add_argument("--clip_gradients", default=1.0, type=float, help="Maximum gradients norm (default: 1.0).")


def load_s2ss_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    add_s2ss_arguments(parser)
    parser = parser.parse_args()
    return parser

