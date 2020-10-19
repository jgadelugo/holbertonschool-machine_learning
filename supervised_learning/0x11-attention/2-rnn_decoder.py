#!/usr/bin/env python3
"""RNNDecoder class to decode for machine translation"""
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """to decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """constructor
        @untis: int representing the number of hidden units in the
        allignement model
        * sets the following public instance attributes
            @embedding: keras Embedding layer that converts words from
            the vocabulary into embedding vector
            @gru: keras GRU layer with units as units
            @F: Dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        @x: tensor of shape(batch, 1) previous word in the target sequence as
        an index of the target vocabulary
        @s_prev: tensor of shape(batch, units) previous decoder hidden state
        @hidden_states: tensor of shape(batch, input_seq_len, units) outputs
        of the encoder
        *can use SelfAttention = __import__('1-self_attention').SelfAttention
        *should concatenate the context vector with x in that order
        Return: y, s
            @y: tensor of shape(batch, vocab) output word as a one hote vector
            in the target vocabulary
            @s: tensor of shape(batch, units) new decoder hidden state
        """
        context, _ = SelfAttention(s_prev.shape[1])(s_prev, hidden_states)

        # pass output sequence thru the input layer
        x = self.embedding(x)
        # concatenate context vector and embedding for output sequence
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, s = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        # pass the output thru F layers
        y = self.F(output)
        return y, s
