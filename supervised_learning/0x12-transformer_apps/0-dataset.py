#!/usr/bin/env python3
""" Class that loads and preps a dataset for ML"""


class Dataset():
    """ Class that loads and preps a dataset for ML"""
    def __init__(self):
        """constructor
        @data_train, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset train split, loaded as_supervided
        @data_valid, which contains the ted_hrlr_translate/pt_to_en
        tf.data.Dataset validate split, loaded as_supervided
        @tokenizer_pt is the Portuguese tokenizer created from the training set
        @tokenizer_en is the English tokenizer created from the training set
        """
        pass
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
        pass
