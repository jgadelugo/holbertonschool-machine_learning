#!/usr/bin/env python3
"""converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow.keras as K


# K.preprocessing.sequence.pad_sequences
def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer
    @model: trained gensim word2vec models
    Return: the trainable keras embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
