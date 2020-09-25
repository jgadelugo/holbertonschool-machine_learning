#!/usr/bin/env python3
"""Function that calculates the scaled dot product attention
source:
https://www.tensorflow.org/tutorials/text/transformer
#scaled_dot_product_attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention
    @Q: tensor with its last two dimensions as (..., seq_len_q, dk) query
    matrix
    @K: tensor with its last two dimensions as (..., seq_len_v, dk) key matrix
    @V: tensor with its last two dimensions as (..., seq_len_v, dv) value
    matrix
    @mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
    optional mask, or defaulted to None
        *if mask is not None, multiply -1e9 to the mask and add it to the
        scaled matrix multiplication
    *The preceding dimensions of Q, K, and V are the same
    Return: output, weights
        @output: tensor with its last two dimensions as (..., seq_len_q, dv)
        scaled dot product attention
        @weights: tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scaled matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)

    # add mask
    if mask is not None:
        logits += (mask * -1e9)

    # softmax normalized on the last axis (se_len_k) so socres add up to 1
    weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
