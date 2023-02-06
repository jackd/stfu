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


class SparseTestCase(tf.test.TestCase):
    def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        a_is_sparse = isinstance(a, tf.SparseTensor)
        b_is_sparse = isinstance(b, tf.SparseTensor)
        if a_is_sparse and b_is_sparse:
            self.assertAllEqual(
                a.dense_shape, b.dense_shape, msg=f"{msg}, dense_shape not equal"
            )
            self.assertAllEqual(a.indices, b.indices, msg=f"{msg}, indices not equal")
            self.assertAllClose(
                a.values,
                b.values,
                rtol=rtol,
                atol=atol,
                msg=f"{msg}, values not allClose",
            )
            return
        if a_is_sparse:
            a = tf.sparse.to_dense(a)
        if b_is_sparse:
            b = tf.sparse.to_dense(b)

        super().assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
