#!/usr/bin/env python3
"""to decode for machine translation"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """to decode for machine translation
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: number of hidden units in the fully connected layer
        @drop_rate: dropout rate
        * sets the following public instance attributes
            @mha1: first MultiHeadAttention
            @mha2: second MultiHeadAttention layer
            @dense_hidden: hidden dense layer with hidden
            @dense_output: the output dense layer with dm units
            @layernorm1: the first layer norm layer, with epsilon=1e-6
            @layernorm2: the second layer norm layer, with epsilon=1e-6
            @layernorm3: the third layer norm layer, with epsilon=1e-6
            @dropout1: the first dropout layer
            @dropout2: the second dropout layer
            @dropout3: the third dropout layer
            ...
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        @x: a tensor of shape (batch, target_seq_len, dm) input to the decoder
        block
        @encoder_output: a tensor of shape (batch, input_seq_len, dm) output
        of the encoder
        @training: a boolean to determine if the model is training
        @look_ahead_mask: the mask to be applied to the first multi head
        attention layer
        @padding_mask: the mask to be applied to the second multi head
        attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm) block’s output
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.droput1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, _ = self.mha2(encoder_output, encoder_output, out1,
                                    padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        hidden = self.dense_hidden(out2)
        out = self.dense_output(hidden)

        ffn_out = self.dropout3(out, training=training)
        out3 = self.layernorm3(ffn_out + out2)

        return out3
