# -*- coding: utf-8 -*-
import argparse
from common_options import *


def add_reg_arguments(parser):
    parser.add_argument("--mode", default="train", help="train, test")
    parser.add_argument("--n_epoch", default=10, help="Max n epoch during training.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size of training.")
    parser.add_argument("--rnn_cell", default='lstm', help='Rnn cell type(rnn, lstm, gru).')
    parser.add_argument("--bidirectional", type=ast.literal_eval, default=True,
                        help="Use bi-directional rnn or not (input should be either 'True' or 'False'.)")
    parser.add_argument("--hidden_size", default=256, help="Hidden size of rnn.")
    parser.add_argument("--num_layers", default=2, help="Rnn layers.")
    parser.add_argument("--keep_prob", default=0.8, type=float, help="Keep prob in dropout.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--clip_gradients", default=3.0, type=float, help="Maximum gradients norm (default: 5.0).")
    parser.add_argument("--optimizer", default="GradientDescentOptimizer",
                        help="The name of the optimizer class in ``tf.train`` or ``tf.contrib.opt`` as a string.")
    parser.add_argument("--sigmoid_pred_score", default=False, type=ast.literal_eval,
                        help="Scale sentiment score from 0 to 1 (input should be either 'True' or 'False'.)")


def load_reg_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    add_reg_arguments(parser)
    return parser.parse_args()

