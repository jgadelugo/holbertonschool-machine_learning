#!/usr/bin/env python3
"""Create a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder
    @input_dims: tuple of ints containing the dimensions of the model input
    @filters: list containing the numbers of filters for each convolutional
    layer in the encoder
        * the filters should be reversed for the decoder
    @latent_dims: tuple of ints, the dimensions of the latent space
    Return: encoder, decoder, auto
        @encoder: the encoder model
        @decoder: decoder model
        @auto: full autoencoder model
    *Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2)
        *The second to last convolution should instead use valid padding
        *The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation
        and no upsampling
    *The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    """
    hl_len = len(filters)
    input_encoder = keras.Input(shape=input_dims)
    input_decoder = keras.Input(shape=latent_dims)

    conv_en = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_encoder)
    pool_en = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        padding="same")(conv_en)

    for i in range(1, hl_len):
        conv_en = keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(pool_en)

        pool_en = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                            padding="same")(conv_en)

    # save encoder model
    encoder = keras.models.Model(inputs=input_encoder, outputs=pool_en)

    conv_de = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_decoder)

    pool_de = keras.layers.UpSampling2D((2, 2))(conv_de)
    for i in range(hl_len - 2, 0, -1):
        conv_de = keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(pool_de)

        pool_de = keras.layers.UpSampling2D((2, 2))(conv_de)

    conv_de = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                  padding='valid', activation='relu')(pool_de)

    pool_de = keras.layers.UpSampling2D((2, 2))(conv_de)

    out = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                              padding='same', activation='sigmoid')(pool_de)

    # save decoder model
    decoder = keras.models.Model(inputs=input_decoder, outputs=out)

    encode = encoder(input_encoder)
    decode = decoder(encode)

    auto = keras.Model(inputs=input_encoder, outputs=decode)
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
