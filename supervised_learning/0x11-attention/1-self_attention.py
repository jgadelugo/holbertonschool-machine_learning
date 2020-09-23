#!/usr/bin/env python3
"""Self attention class calculates the attention for machine translation
based on https://arxiv.org/pdf/1409.0473.pdf
Other resource used
https://towardsdatascience.com/implementing-neural-machine-translation-
with-attention-using-tensorflow-fc9c6f26155f
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculate the attention for machine translation based on
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, units):
        """constructor
        @untis: int representing the number of hidden units in the
        allignement model
        * sets the following public instance attributes
            @W: Dense layer with units as units, to be applied to the previous
            decoder hidden state
            @U: Dense layer with units as units, to be applied to the encoder
            hidden states
            @V: Dense layer with 1 units, to be applied to the tanh of the sum
            of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        @s_prev: tensor of shape(batch, units) previous decoder hidden state
        @hidden_states: tensor of shape(batch, input_seq_len, units) outputs
        of the encoder
        Return: context, weights
            @context: tensor of shape(batch, units) the context vector for the
            decoder
            @weights: tensor of shape(batch, input_seq_len, 1) the attention
            weights
        """
        query = tf.expand_dims(s_prev, 1)

        # Calculate the attention score
        score = self.V(tf.nn.tanh(self.W(query) + self.U(hidden_states)))
        # weights
        weights = tf.nn.softmax(score, axis=1)
        # context vector for decoder
        context_vector = weights * hidden_states
        context = tf.reduce_sum(context_vector, axis=1)

        return context, weights
