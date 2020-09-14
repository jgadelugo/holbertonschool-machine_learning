#!/usr/bin/env python3
"""creates a bag of words embedding matrix"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix
    @sentences: list of sentences to analyze
    @vocab: list of vocabulary words to use for the analysis
        *if None, all words within sentences should be used
    Return: embeddings, features
        @embeddings: np.ndarray shape(s, f) containing the embeddings
            @s: number of sentences in sentences
            @f: number of features analyzed
        @features: list of features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()

    features = vectorizer.get_feature_names()

    return embeddings, features
