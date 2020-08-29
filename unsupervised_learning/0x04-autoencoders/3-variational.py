#!/usr/bin/env python3
"""Create an autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a sparse autoencoder
    @input_dims: an int containing the dimensions of the model input
    @hidden_layers: list containing the numbers of nodes for each hidden layer
    in the encoder
        * the hidden layers should be reversed for the decoder
    @latent_dims: an int, the dimensions of the latent space representation
    Return: encoder, decoder, auto
        @encoder: the encoder model
        @decoder: decoder model
        @auto: full autoencoder model
    """
    hl_len = len(hidden_layers)
    input_encoder = keras.Input(shape=(input_dims,))
    input_decoder = keras.Input(shape=(latent_dims,))

    enco = keras.layers.Dense(units=hidden_layers[0],
                              activation='relu')(input_encoder)

    for i in range(1, hl_len):
        enco = keras.layers.Dense(units=hidden_layers[i],
                                  activation='relu')(enco)
    f_encoded = keras.layers.Dense(units=latent_dims, activation='relu')(enco)
    # save encoder model
    encoder = keras.models.Model(inputs=input_encoder, outputs=f_encoded)

    decoded = keras.layers.Dense(units=hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for i in range(hl_len - 2, -1, -1):
        decoded = keras.layers.Dense(units=hidden_layers[i],
                                     activation='relu')(decoded)
    f_decoded = keras.layers.Dense(units=input_dims,
                                   activation='sigmoid')(decoded)

    # save decoder model
    decoder = keras.models.Model(inputs=input_decoder, outputs=f_decoded)

    _input = keras.Input(shape=(input_dims,))
    encode = encoder(_input)
    decode = decoder(encode)

    auto = keras.Model(inputs=_input, outputs=decode)
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
