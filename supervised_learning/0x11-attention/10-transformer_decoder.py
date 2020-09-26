#!/usr/bin/env python3
""" Decoder for a transformer"""
import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding


class Decoder(tf.keras.layers.Layer):
    """to Decoder for machine translation
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor
        @N: number of blocks in encoder
        @dm: dimensionality of model
        @h: number of heads
        @hidden: number of hidden units in fully connected layer
        @target_vocab: size of target vocabulary
        @max_seq_len: maximum sequence length possible
        @drop_rate: dropout rate
        *Sets following public instance attributes:
            @N: number of blocks in encoder
            @dm: dimensionality of model
            @embedding: embedding layer for inputs
            @positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
            containing positional encodings
            @blocks: a list of length N containing all of EncoderBlock‘s
            @dropout: dropout layer, to be applied to positional encodings
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        @x: tensor of shape (batch, target_seq_len, dm) input to decoder
        @encoder_output: tensor of shape (batch, input_seq_len, dm) output of
        encoder
        @training: boolean to determine if model is training
        @look_ahead_mask: mask to be applied to first multi head attention
        layer
        @padding_mask: mask to be applied to second multi head attention layer
        Returns: tensor of shape (batch, target_seq_len, dm) decoder output
        """
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, _, _ = self.blocks[i](x, encoder_output, training,
                                     look_ahead_mask, padding_mask)
        return x
