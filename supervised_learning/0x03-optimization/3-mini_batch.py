#!/usr/bin/env python3
""" train using mini-batch gradient descent"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, b_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """ train using mini-batch gradient descent
    @X_train: numpy.ndarray of shape (m, 784) containing training data
        @m: number of data points
        @784: number of input features
    @Y_train: is a one-hot numpy.ndarray shape (m, 10) to training labels
        @10: number of classes model should classify
    @X_valid: numpy.ndarray of shape (m, 784) containing validation data
    @Y_valid: is one-hot numpy.ndarray shape (m, 10) validation labels
    @b_size: is the number of data points in a batch
    @epochs: number of times training should pass thorugh dataset
    @load_path: path to where the model should be loaded
    @save_path: path to where the model should be saved
    Returns: the path where the model was saved
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        size = X_train.shape[0]
        inner = size // b_size if size % b_size == 0 else size // b_size + 1

        for i in range(epochs + 1):
            # shuffle to avoid over fitting
            X_s, Y_s = shuffle_data(X_train, Y_train)

            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(t_cost))
            print('\tTraining Accuracy: {}'.format(t_acc))
            print('\tValidation Cost: {}'.format(v_cost))
            print('\tValidation Accuracy: {}'.format(v_acc))

            if i < epochs:
                end = b_size
                for j in range(inner):
                    start = j * b_size
                    end = (j + 1) * b_size if end <= size else size
                    feed = {x: X_s[start: end], y: Y_s[start: end]}
                    sess.run(train_op, feed_dict=feed)
                    if (j + 1) % 100 == 0 and j != 0:
                        c_cost = sess.run(loss, feed_dict=feed)
                        c_acc = sess.run(accuracy, feed_dict=feed)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(c_cost))
                        print("\t\tAccuracy: {}".format(c_acc))

        save_path = saver.save(sess, save_path)
    return save_path
