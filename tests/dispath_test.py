import tensorflow as tf

from stfu import dispatch, test_utils

del dispatch


class DispatchTest(test_utils.SparseTestCase):
    def test_neg(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        self.assertAllClose(-st, st.with_values(-st.values))

    def test_abs(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        self.assertAllClose(abs(st), st.with_values(tf.abs(st.values)))

    def test_conj(self):
        st = test_utils.random_sparse(5, (3, 4))
        self.assertAllClose(tf.math.conj(st), tf.sparse.map_values(tf.math.conj, st))

    def test_transpose(self):
        st = test_utils.random_sparse(5, (3, 4))
        self.assertAllClose(tf.transpose(st), tf.sparse.transpose(st))

    def test_add_sparse_sparse(self):
        shape = (3, 4)
        st0 = test_utils.random_sparse(5, shape)
        st1 = test_utils.random_sparse(4, shape)
        self.assertAllClose(st0 + st1, tf.sparse.add(st0, st1))

    def test_add_sparse_dense(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(st + dense, tf.sparse.to_dense(st) + dense)
        dense = dense[0]
        self.assertAllClose(st + dense, tf.sparse.to_dense(st) + dense)

    def test_add_dense_sparse(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(tf.math.add(dense, st), dense + tf.sparse.to_dense(st))
        dense = dense[0]
        self.assertAllClose(tf.math.add(dense, st), dense + tf.sparse.to_dense(st))

    def test_subtract_sparse_sparse(self):
        shape = (3, 4)
        st0 = test_utils.random_sparse(5, shape)
        st1 = test_utils.random_sparse(4, shape)
        self.assertAllClose(st0 - st1, tf.sparse.add(st0, st1.with_values(-st1.values)))

    def test_subtract_sparse_dense(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(st - dense, tf.sparse.to_dense(st) - dense)

    def test_subtract_dense_sparse(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(tf.math.subtract(dense, st), dense - tf.sparse.to_dense(st))

    def test_multiply_sparse_dense(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(
            tf.sparse.to_dense(st * dense), tf.sparse.to_dense(st) * dense
        )
        self.assertAllClose(
            tf.sparse.to_dense(st * dense[0]), tf.sparse.to_dense(st) * dense[0]
        )

    def test_multiply_dense_sparse(self):
        shape = (3, 4)
        st = test_utils.random_sparse(5, shape)
        dense = tf.random.normal(shape)
        self.assertAllClose(
            tf.sparse.to_dense(tf.math.multiply(dense, st)),
            tf.sparse.to_dense(st) * dense,
        )
        self.assertAllClose(
            tf.sparse.to_dense(tf.math.multiply(dense[0], st)),
            tf.sparse.to_dense(st) * dense[0],
        )

    def test_matmul(self):
        n, m, p = (5, 4, 7)
        a = test_utils.random_sparse(5, (n, m))
        b = tf.random.normal((m, p))
        bt = tf.transpose(b)
        at = tf.sparse.transpose(a)

        self.assertAllClose(tf.linalg.matmul(a, b), tf.sparse.sparse_dense_matmul(a, b))
        self.assertAllClose(a @ b, tf.sparse.sparse_dense_matmul(a, b))
        self.assertAllClose(
            tf.linalg.matmul(at, b, transpose_a=True),
            tf.sparse.sparse_dense_matmul(at, b, adjoint_a=True),
        )
        self.assertAllClose(
            tf.linalg.matmul(at, b, adjoint_a=True),
            tf.sparse.sparse_dense_matmul(at, b, adjoint_a=True),
        )
        self.assertAllClose(
            tf.linalg.matmul(a, bt, transpose_b=True),
            tf.sparse.sparse_dense_matmul(a, bt, adjoint_b=True),
        )
        self.assertAllClose(
            tf.linalg.matmul(a, bt, adjoint_b=True),
            tf.sparse.sparse_dense_matmul(a, bt, adjoint_b=True),
        )


if __name__ == "__main__":
    tf.test.main()
