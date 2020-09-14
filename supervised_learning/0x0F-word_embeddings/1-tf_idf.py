#!/usr/bin/env python3
""" creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ creates a TF-IDF embedding
    @sentences: list of sentences to analyze
    @vocab: list of vocabulary words to use for the analysis
        *if None, all words within sentences should be used
    Return: embeddings, features
        @embeddings: np.ndarray shape(s, f) containing the embeddings
            @s: number of sentences in sentences
            @f: number of features analyzed
        @features: list of features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()

    features = vectorizer.get_feature_names()

    return embeddings, features
