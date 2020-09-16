#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence
    @references: list of reference translations
        *each reference translation is a list of the words in the translation
    @sentence: list containing the model proposed sentence
    Return: the unigram BLUE score
    """
    # calculates c and r
    c = len(sentence)
    r = min([len(ref) for ref in references])

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

    if c <= r:
        brevity_penalty = np.exp(1 - r / c)
    else:
        brevity_penalty = 1
    return brevity_penalty * count_appear / c
