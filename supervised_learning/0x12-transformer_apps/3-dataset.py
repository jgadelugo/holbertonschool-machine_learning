#!/usr/bin/env python3
""" Class that loads and preps a dataset for ML
source: https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Class that loads and preps a dataset for ML"""
    def __init__(self, batch_size, max_len):
        """constructor
        * create instance attributes:
        @data_train, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset train split, loaded as_supervided
        @data_valid, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset validate split, loaded as_supervided
        @tokenizer_pt: Portuguese tokenizer created from the training set
        @tokenizer_en: English tokenizer created from the training set
        """

        def filter_max_length(x, y, max_length=max_len):
            """ filter max len"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()

        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset
        @data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
            @pt: tf.Tensor Portuguese sentence
            @en: tf.Tensor corresponding English sentence
        *max vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
            @tokenizer_pt: Portugues tokenizer
            @tokenizer_en: English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens
        @pt: tf.Tensor containing the Portuguese sentence
        @en: tf.Tensor containing the corresponding English sentence
        *tokenized sentences should include start and end of sentence tokens
        *start token should be indexed as vocab_size
        *end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """ acts as a tensorflow wrapper for the encode instance method
        Make sure to set the shape of the pt and en return tensors
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
