#!/usr/bin/env python3
""" Class that loads and preps a dataset for ML
source: https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Class that loads and preps a dataset for ML"""
    def __init__(self):
        """constructor
        * create instance attributes:
        @data_train, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset train split, loaded as_supervided
        @data_valid, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset validate split, loaded as_supervided
        @tokenizer_pt: Portuguese tokenizer created from the training set
        @tokenizer_en: English tokenizer created from the training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

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
