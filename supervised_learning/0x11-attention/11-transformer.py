#!/usr/bin/env python3
""" Transformer for machine translation"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """Transformer for machine translation
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """constructor
        @N: number of blocks in encoder and decoder
        @dm: dimensionality of model
        @h: number of heads
        @hidden: number of hidden units in fully connected layers
        @input_vocab: size of input vocabulary
        @target_vocab: size of target vocabulary
        @max_seq_input: maximum sequence length possible for input
        @max_seq_target: maximum sequence length possible for target
        @drop_rate: dropout rate
        *Sets following public instance attributes:
            @encoder: encoder layer
            @decoder: decoder layer
            @linear: final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, N, h, hidden,
                               input_vocab, max_seq_input, drop_rate)

        self.decoder = Decoder(N, N, h, hidden,
                               target_vocab, max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        @inputs: tensor of shape (batch, input_seq_len, dm) inputs
        @target: tensor of shape (batch, target_seq_len, dm) target
        @training: boolean to determine if model is training
        @encoder_mask: padding mask to be applied to encoder
        @look_ahead_mask: look ahead mask to be applied to decoder
        @decoder_mask: padding mask to be applied to decoder
        Returns: tensor of shape (batch, target_seq_len, target_vocab)
        transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, _ = self.decoder(target, enc_output, training,
                                     look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
