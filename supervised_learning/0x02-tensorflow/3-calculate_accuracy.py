#!/usr/bin/env python3
"""calculates the accuracy of a prediction"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    labels = tf.argmax(y, 1)
    predictions = tf.argmax(y_pred, 1)
    correct = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
