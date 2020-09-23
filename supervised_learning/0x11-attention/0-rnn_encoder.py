#!/usr/bin/env python3
"""RNNEncoder to encode for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """To encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """constructor
        @vocab: int representing the size of the input vocabulary
        @embedding: int representing the dimensionality of the embedding
        vector
        @untis: int representing the number of hidden units in the RNN cell
        @bath: int representing the batch size
        * sets the following public instance attributes
            @batch: batch size
            @units: number of hidden units in the RNN cell
            @embedding: keras Embedding layer that converts words from the
            vacabulary into an embedding vector
            @gru: keras GRU layer with units as units
                *should return both the full sequence of outputs as well
                as the last hidden state
                * Recurrent weights should be initialized with glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """initializes the hidden states for the RNN cell to a tensor of zeros
        Return: tensor of shape(batch, units) containing the initialized
        hidden states
        """
        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        @x: tensor of shape(batch, input_seq_len) input to the encoder layer
        as word indices within the vocabulary
        @initial: tensor of shape (batch, units) initial hidden state
        Return: outputs, hidden
            @outputs tensor of shape (batch, input_seq_len, units) outputs of
            the encoder
            @hidden: tensor of shape(batch, units) last hidden state of the
            encoder
        """
        outputs, hidden = self.gru(inputs=self.embedding(x),
                                   initial_state=initial)
        return outputs, hidden
