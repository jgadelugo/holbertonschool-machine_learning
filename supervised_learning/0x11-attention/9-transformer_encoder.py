#!/usr/bin/env python3
""" encoder for a transformer"""
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding


class Encoder(tf.keras.layers.Layer):
    """to encoder for machine translation
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor
        @N: number of blocks in encoder
        @dm: dimensionality of model
        @h: number of heads
        @hidden: number of hidden units in fully connected layer
        @input_vocab: size of input vocabulary
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
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       self.dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        @x: tensor of shape (batch, input_seq_len, dm) input to the encoder
        @training: boolean to determine if the model is training
        @mask: mask to be applied for multi head attention
        Returns: tensor of shape (batch, input_seq_len, dm) encoder output
        """
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.enc_layers[i](x, training, mask)

        return x
