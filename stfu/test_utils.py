import typing as tp

import tensorflow as tf


def random_sparse(nnz: int, shape: tp.Sequence[int]) -> tf.SparseTensor:
    """
    Get a random sparse tensor with at most `nnz` non-zeros.

    Args:

    """
    size = tf.reduce_prod(shape)
    indices = tf.random.uniform((nnz,), maxval=tf.cast(size, tf.int64), dtype=tf.int64)
    indices, _ = tf.unique(indices)
    indices = tf.cast(tf.transpose(tf.unravel_index(indices, shape), (1, 0)), tf.int64)
    values = tf.random.normal(shape=(tf.shape(indices)[0],))
    return tf.sparse.reorder(tf.SparseTensor(indices, values, shape))
