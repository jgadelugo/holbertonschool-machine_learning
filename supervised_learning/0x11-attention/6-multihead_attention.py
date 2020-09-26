#!/usr/bin/env python3
"""MultiHeadAttention class to perform multihead attention
Other resource used
https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """perform multihead attention
    """
    def __init__(self, dm, h):
        """constructor
        @dm: an integer representing the dimensionality of the model
        @h: an integer representing the number of heads
        @dm: divisible by h
        *Sets the following public instance attributes:
            @h: the number of heads
            @dm: the dimensionality of the model
            @depth: the depth of each attention head
            @Wq: Dense layer with dm units, used to generate the query matrix
            @Wk: Dense layer with dm units, used to generate the key matrix
            @Wv: Dense layer with dm units, used to generate the value matrix
            @linear: Dense layer with dm units, used to generate the attention
            output
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)
        @x: tensor input, query, key, or value
        @batch_size: batch size
        Return: result shape(batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        @Q: tensor of shape (batch, seq_len_q, dk) input to generate the query
        matrix
        @K: tensor of shape (batch, seq_len_v, dk) input to generate the key
        matrix
        @V: tensor of shape (batch, seq_len_v, dv) input to generate the value
        matrix
        *mask: always None
        Return: output, weights
            @output: tensor with its last two dimensions as
            (..., seq_len_q, dm) scaled dot product attention
            @weights: tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) attention weights
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        att_out, weights = sdp_attention(q, k, v, mask)

        att_out = tf.transpose(att_out, perm=[0, 2, 1, 3])
        con_att = tf.reshape(att_out, (batch_size, -1, self.dm))

        output = self.linear(con_att)

        return output, weights
