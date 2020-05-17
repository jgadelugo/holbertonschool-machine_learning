#!/usr/bin/env python3
"""builds, trains, and saves a neural network classifier"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier
    @X_train: numpy.ndarray - training input data
    @Y_train: numpy.ndarray - training labels
    @X_valid: numpy.ndarray - Validation input data
    @Y_valid: numpy.ndarray - Validation labels
    @layer_sizes: list of number of nodes in each layer
    @actications: list of activation functions
    @alpha: learning rate
    @iterations: number of iterations to train over
    @save_path: path to save the model
    Returns: the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    acc = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('t_acc', acc)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        t_loss, t_acc = sess.run((loss, acc), feed_dict={x: X_train,
                                                         y: Y_train})
        v_loss, v_acc = sess.run((loss, acc), feed_dict={x: X_valid,
                                                         y: Y_valid})
        i = 0
        while i <= iterations:
            if (i % 100 == 0) or (i == iterations):
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_loss))
                print("\tTraining t_acc: {}".format(t_acc))
                print("\tValidation Cost: {}".format(v_loss))
                print("\tValidation t_acc: {}".format(v_acc))
            if (i < iterations):
                sess.run((train_op), feed_dict={x: X_train, y: Y_train})
                t_loss, t_acc = sess.run((loss, acc), feed_dict={x: X_train,
                                                                 y: Y_train})
                v_loss, v_acc = sess.run((loss, acc), feed_dict={x: X_valid,
                                                                 y: Y_valid})
            i += 1
        save_path = saver.save(sess, save_path)
    return save_path
