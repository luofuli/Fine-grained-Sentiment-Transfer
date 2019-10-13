# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2018/7/30 11:41 
"""

import tensorflow as tf
from utils import constants


def load_paired_dataset(input_files,
                        input_vocab,
                        batch_size=32,
                        min_seq_len=3,
                        max_seq_len=25):
    """Returns an iterator over the training data."""

    def _make_dataset(text_file, input_vocab):
        def decode_csv(line):
            parsed_line = tf.decode_csv(line, [[''], [0.], [''], [0.]], field_delim='\t')
            x1 = tf.string_split([parsed_line[0]]).values
            s1 = parsed_line[1]
            x2 = tf.string_split([parsed_line[2]]).values
            s2 = parsed_line[3]

            bos = tf.constant([constants.START_OF_SENTENCE_TOKEN], dtype=tf.string)
            eos = tf.constant([constants.END_OF_SENTENCE_TOKEN], dtype=tf.string)
            d = {"source_ids": input_vocab.lookup(x1),
                 "source_ids_in": input_vocab.lookup(tf.concat([bos, x1], axis=0)),
                 "source_ids_out": input_vocab.lookup(tf.concat([x1, eos], axis=0)),
                 "source_length": tf.shape(x1)[0],
                 "source_senti": s1,
                 "target_ids": input_vocab.lookup(x2),
                 "target_ids_in": input_vocab.lookup(tf.concat([bos, x2], axis=0)),
                 "target_ids_out": input_vocab.lookup(tf.concat([x2, eos], axis=0)),
                 "target_length": tf.shape(x2)[0],
                 "target_senti": s2}
            return d

        dataset = tf.data.TextLineDataset(text_file)
        dataset = dataset.map(decode_csv)
        return dataset

    # Make a dataset from the input and translated file.
    input_dataset = _make_dataset(input_files, input_vocab)
    dataset = input_dataset.shuffle(100000)  # 200000

    # Filter out invalid examples.
    dataset = dataset.filter(lambda x: tf.greater(x["source_length"], min_seq_len - 1))
    dataset = dataset.filter(lambda x: tf.greater(x["target_length"], min_seq_len - 1))

    # Batch and pad the dataset
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes={
                                       "source_ids": [tf.Dimension(max_seq_len)],  # size is unknown
                                       "source_ids_in": [tf.Dimension(max_seq_len)],  # size is unknown
                                       "source_ids_out": [tf.Dimension(max_seq_len)],  # size is unknown
                                       "source_length": [],  # don't need pad
                                       "source_senti": [],
                                       "target_ids": [tf.Dimension(max_seq_len)],
                                       "target_ids_in": [tf.Dimension(max_seq_len)],
                                       "target_ids_out": [tf.Dimension(max_seq_len)],
                                       "target_length": [],
                                       "target_senti": []},
                                   # padding_values=constants.PADDING_ID,
                                   )
    return dataset.make_initializable_iterator()


def load_dataset(input_file,
                 input_vocab,
                 mode,
                 batch_size=32,
                 min_seq_len=3,
                 max_seq_len=25,
                 has_source=False):
    """Returns an iterator over the training data."""
    def _make_dataset(text_file, vocab):
        def decode_csv(line):
            if mode in [constants.TEST, constants.INFER]:  # test file does not conclude labels.
                x = tf.string_split([line]).values
                s = tf.constant(0)  # a fixed value
            else:
                if has_source:
                    parsed_line = tf.decode_csv(line, [[''], [''], [0.]], field_delim='\t')
                    x = tf.string_split([parsed_line[1]]).values
                    s = parsed_line[2]
                else:
                    parsed_line = tf.decode_csv(line, [[''], [0.]], field_delim='\t')
                    x = tf.string_split([parsed_line[0]]).values
                    s = parsed_line[1]

            bos = tf.constant([constants.START_OF_SENTENCE_TOKEN], dtype=tf.string)
            eos = tf.constant([constants.END_OF_SENTENCE_TOKEN], dtype=tf.string)
            d = {"ids": vocab.lookup(x),
                 "ids_in": vocab.lookup(tf.concat([bos, x], axis=0)),
                 "ids_out": vocab.lookup(tf.concat([x, eos], axis=0)),
                 "ids_in_out": vocab.lookup(tf.concat([bos, x, eos], axis=0)),
                 "length": tf.shape(x)[0],
                 "senti": s,
                 }
            return d

        dataset = tf.data.TextLineDataset(text_file)
        dataset = dataset.map(decode_csv)
        return dataset

    # Make a dataset from the input and translated file.
    dataset = _make_dataset(input_file, input_vocab)
    if mode == constants.TRAIN:
        dataset = dataset.shuffle(200000)

    # Filter out invalid examples.
    if mode == constants.TRAIN:
        dataset = dataset.filter(lambda x: tf.greater(x["length"], min_seq_len - 1))

    # Batch the dataset using a bucketing strategy.
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes={
                                       "ids": [tf.Dimension(max_seq_len)],
                                       "ids_in": [tf.Dimension(max_seq_len)],
                                       "ids_out": [tf.Dimension(max_seq_len)],
                                       "ids_in_out": [tf.Dimension(max_seq_len)],
                                       "length": [],  # don't need pad
                                       "senti": []},
                                   )
    return dataset.make_initializable_iterator()


def check_dataset(path, max_seq_len=30):
    for i, line in enumerate(open(path)):
        words = line.strip().split(' ')

        if len(words) > max_seq_len-2:
            print(i, line)
            return False
    return True


if __name__ == "__main__":
    check_dataset("../data/yelp/test.")