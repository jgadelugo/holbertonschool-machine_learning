#!/usr/bin/env python3
"""Transformer"""
import tensorflow.compat.v2 as tf
import numpy as np


def point_wise_feed_forward_network(dm, hidden):
    """feed forward"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dense(dm)])


def get_angles(pos, i, d_model):
    """get angles"""
    angle_rates = 1 / np.power(10000, (2 @(i // 2)) / np.float32(d_model))
    return pos @angle_rates


def positional_encoding(position, d_model):
    """positional encoding"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def sdp_attention(Q, K, V, mask=None):
    """
    @Q: tensor with its last two dimensions as (..., seq_len_q, dk)
    containing the query matrix
    @K: tensor with its last two dimensions as (..., seq_len_v, dk)
    containing the key matrix
    @V: tensor with its last two dimensions as (..., seq_len_v, dv)
    containing the value matrix
    @mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
    containing the optional mask, or defaulted to None
    @if mask: not None, multiply -1e9 to the mask and add it to the scaled
    matrix multiplication
    @The preceding dimensions of Q, K, and V are the same
    Returns: output, weights
        @outputa tensor with its last two dimensions as
        (..., seq_len_q, dv) containing the scaled dot product attention
        @weights a tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) containing the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi Head Attention Class"""
    def __init__(self, dm, h):
        """
        @dm:n integer representing the dimensionality of the model
        @h:n integer representing the number of heads
        @dm is divisible by h
        @Sets the following public instance attributes:
        @h: the number of heads
        @dm: the dimensionality of the model
        @depth: the depth of each attention head
        @Wq: Dense layer with dm units, used to generate the query matrix
        @Wk: Dense layer with dm units, used to generate the key matrix
        @Wv: Dense layer with dm units, used to generate the value matrix
        @linear: Dense layer with dm units, used to generate the
          attention output
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
        """
        @Split the last dimension into (num_heads, depth).
        @Transpose the result such that the shape is
          (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        @Q: tensor of shape (batch, seq_len_q, dk) containing the
          input to generate the query matrix
        @K: tensor of shape (batch, seq_len_v, dk) containing the
          input to generate the key matrix
        @V: tensor of shape (batch, seq_len_v, dv) containing the
          input to generate the value matrix
        @mask:lways None
        Returns: output, weights
        @outputa tensor with its last two dimensions as (..., seq_len_q, dm)
          containing the scaled dot product attention
        @weights a tensor with its last three dimensions as
         (..., h, seq_len_q, seq_len_v) containing the attention weights
        """

        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """class Encoder block"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully connected layer
        @drop_rate: the dropout rate
        Sets the following public instance attributes:
        @mha: MultiHeadAttention layer
        @dense_hidden: the hidden dense layer with hidden units an
          relu activation
        @dense_output: the output dense layer with dm units
        @layernorm1: the first layer norm layer, with epsilon=1e-6
        @layernorm2: the second layer norm layer, with epsilon=1e-6
        @dropout1: the first dropout layer
        @dropout2: the second dropout layer
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        @x: tensor of shape (batch, input_seq_len, dm)containing
          the input to the encoder block
        @training: boolean to determine if the model is training
        @mask: the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm) containing
        the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        final_output = self.layernorm2(out1 + ffn_output)

        return final_output


class DecoderBlock(tf.keras.layers.Layer):
    """class DecoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Constructor
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully
          connected layer
        @drop_rate: the dropout rate
        Sets the following public instance attributes:
        @mha1: the first MultiHeadAttention layer
        @mha2: the second MultiHeadAttention layer
        @dense_hidden: the hidden dense layer with hidden
          units and relu activation
        @dense_output: the output dense layer with dm units
        @layernorm1: the first layer norm layer, with epsilon=1e-6
        @layernorm2: the second layer norm layer, with epsilon=1e-6
        @layernorm3: the third layer norm layer, with epsilon=1e-6
        @dropout1: the first dropout layer
        @dropout2: the second dropout layer
        @dropout3: the third dropout layer
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        @x: tensor of shape (batch, target_seq_len, dm)containing the
          input to the decoder block
        @encoder_output: tensor of shape (batch, input_seq_len, dm)
          containing the output of the encoder
        @training: boolean to determine if the model is training
        @look_ahead_mask: the mask to be applied to the first multi head
          attention layer
        @padding_mask: the mask to be applied to the second multi
          head attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm) containing
        the block’s output
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)

        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)

        ffn_output = self.dropout3(ffn_output, training=training)
        output_final = self.layernorm3(ffn_output + out2)

        return output_final, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """Encoder class"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Constructor
        @N: the number of blocks in the encoder
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the
          fully connected layer
        @input_vocab: the size of the input vocabulary
        @max_seq_len: the maximum sequence length possible
        @drop_rate: the dropout rate
        Sets the following public instance attributes:
        @N: the number of blocks in the encoder
        @dm: the dimensionality of the model
        @embedding: the embedding layer for the inputs
        @positional_encoding: numpy.ndarray of shape
          (max_seq_len, dm) containing the positional encodings
        @blocks: list of length N containing all of the
          EncoderBlock‘s
        @dropout: the dropout layer, to be applied to the
          positional encodings
        """
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        @x: tensor of shape (batch, input_seq_len, dm)containing
          the input to the encoder
        @training: boolean to determine if the model is training
        @mask: the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm) containing
        the encoder output
        """
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        embedding = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:, :seq_len, :]

        encoder_out = self.dropout(embedding, training=training)

        for i in range(self.N):
            encoder_out = self.blocks[i](encoder_out, training, mask)

        return encoder_out


class Decoder(tf.keras.layers.Layer):
    """class Decoder"""
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """ Constructor
        @N: the number of blocks in the encoder
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully
          connected layer
        @target_vocab: the size of the target vocabulary
        @max_seq_len: the maximum sequence length possible
        @drop_rate: the dropout rate
        Sets the following public instance attributes:
        @N: the number of blocks in the encoder
        @dm: the dimensionality of the model
        @embedding: the embedding layer for the targets
        @positional_encoding: numpy.ndarray of shape
          (max_seq_len, dm) containing the positional encodings
        @blocks: list of length N containing all of the
          DecoderBlock‘s
        @dropout: the dropout layer, to be applied to the
          positional encodings
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        @x: tensor of shape (batch, target_seq_len, dm)containing
          the input to the decoder
        @encoder_output: tensor of shape (batch, input_seq_len, dm)
          containing the output of the encoder
        @training: boolean to determine if the model is training
        @look_ahead_mask: the mask to be applied to the first multi
          head attention layer
        @padding_mask: the mask to be applied to the second multi
          head attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm)
        containing the decoder output
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, block1, block2 = self.blocks[i](x, encoder_output, training,
                                               look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights


class Transformer(tf.keras.Model):
    """class Transform"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ Constructor
        @N: the number of blocks in the encoder and decoder
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully
          connected layers
        @input_vocab: the size of the input vocabulary
        @target_vocab: the size of the target vocabulary
        @max_seq_input: the maximum sequence length possible
          for the input
        @max_seq_target: the maximum sequence length possible
          for the target
        @drop_rate: the dropout rate
        Sets the following public instance attributes:
        @encoder: the encoder layer
        @decoder: the decoder layer
        @linear: final Dense layer with target_vocab units
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        @inputs: tensor of shape (batch, input_seq_len, dm)
          containing the inputs
        @target: tensor of shape (batch, target_seq_len, dm)
          containing the target
        @training: boolean to determine if the model is training
        @encoder_mask: the padding mask to be applied to the encoder
        @look_ahead_mask: the look ahead mask to be applied to the decoder
        @decoder_mask: the padding mask to be applied to the decoder
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        containing the transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output, attention = self.decoder(target, enc_output, training,
                                             look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output, attention
