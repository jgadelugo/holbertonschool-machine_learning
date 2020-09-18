#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence"""
import numpy as np


def grab_ngram(sentence, n):
    """creates new list of strings for ngram"""
    size = len(sentence)
    sent_ngram = []
    for i in range(size - n + 1):
        sent_ngram.append("".join([sentence[j] for j in range(i, n+i)]))
    return sent_ngram


def get_precision(references, sentence, n):
    """Calculates the unigram BLEU score for a sentence
    @references: list of reference translations
        *each reference translation is a list of the words in the translation
    @sentence: list containing the model proposed sentence
    @n: size of the n-gram to use for evaluation
    Return: the n-gram BLUE score
    """
    r = min([len(ref) for ref in references])

    # modify strings to get ngrams
    sentence = grab_ngram(sentence, n)
    references = [grab_ngram(ref, n) for ref in references]

    # new c for precision
    new_c = len(sentence)

    # list of word count in each reference
    refs = [{x: ref.count(x) for x in ref} for ref in references]
    # word count in sentence
    word_count = {x: sentence.count(x) for x in sentence}

    ref_count = {}
    for ref in refs:
        for key in ref.keys():
            if key not in ref_count or ref[key] > ref_count[key]:
                ref_count[key] = ref[key]

    count_appear = 0
    for word in word_count.keys():
        if word in ref_count.keys():
            count_appear += min(word_count[word], ref_count[word])

    precision = count_appear / new_c

    return precision


def geo_mean_overflow(iterable):
    """ Calculate the geometric mean"""
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))


def cumulative_bleu(references, sentence, n):
    """Calculates the unigram BLEU score for a sentence
    @references: list of reference translations
        *each reference translation is a list of the words in the translation
    @sentence: list containing the model proposed sentence
    @n: size of the n-gram to use for evaluation
    Return: the n-gram BLUE score
    """
    # calculates c and r
    c = len(sentence)
    r = min([len(ref) for ref in references])

    precisions = []
    for i in range(1, n+1):
        precisions.append(get_precision(references, sentence, i))

    precision = geo_mean_overflow(precisions)

    if c <= r:
        brevity_penalty = np.exp(1 - r / c)
    else:
        brevity_penalty = 1

    BLEU_score = brevity_penalty * precision
    return BLEU_score
