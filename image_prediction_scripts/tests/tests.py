
import tensorflow as tf


def test_multipication():
    vector = tf.constant([1, 2, 1])
    matrix = tf.constant([[1, 2], [1,2], [1,2]])
    expanded_vector = tf.expand_dims(vector, axis=1)
    result = tf.multiply(matrix, expanded_vector)

    print(result)

test_multipication()
