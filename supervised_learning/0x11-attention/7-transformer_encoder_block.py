#!/usr/bin/env python3
"""EncoderBlock Create encoder block for a transformer
Other resource used
https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor
        @dm: dimensionality of the model
        @h: number of heads
        @hidden: number of hidden units in the fully connected layer
        @drop_rate: dropout rate
        *Sets the following public instance attributes:
            @mha: a MultiHeadAttention layer
            @dense_hidden: hidden dense layer with hidden units and relu
            activation
            @dense_output: output dense layer with dm units
            @layernorm1: first layer norm layer, with epsilon=1e-6
            @layernorm2: second layer norm layer, with epsilon=1e-6
            @dropout1: first dropout layer
            @dropout2: second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        @x: tensor shape (batch, input_seq_len, dm) input to the encoder block
        @training: boolean to determine if the model is training
        @mask: mask to be applied for multi head attention
        Return: tensor shape (batch, input_seq_len, dm) the block’s output
        """
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(x + attn_out)

        hidden_out = self.dense_hidden(out1)
        out = self.dense_output(hidden_out)

        fnn_out = self.dropout2(out, training=training)

        out2 = self.layernorm2(out1, fnn_out)

        return out2
