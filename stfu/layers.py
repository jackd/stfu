import tensorflow as tf


class SparseTensorLayer(tf.keras.layers.Layer):
    def call(self, args):
        return tf.SparseTensor(*args)


def sparse_tensor(indices: tf.Tensor, values: tf.Tensor, dense_shape: tf.Tensor):
    return SparseTensorLayer()((indices, values, dense_shape))
