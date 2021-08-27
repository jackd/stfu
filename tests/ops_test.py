import tensorflow as tf
from absl.testing import parameterized

from stfu import ops
from stfu.test_utils import random_sparse


class OpsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters([None, 0, 2, ((0, 2),), -1, -3, ((2, 0),), ((0, -1),)])
    def test_squeeze(self, axis):
        value = random_sparse(10, (1, 12, 1))
        expected = tf.squeeze(tf.sparse.to_dense(value), axis=axis)
        actual = tf.sparse.to_dense(ops.squeeze(value, axis=axis))
        self.assertAllClose(actual, expected)

    @parameterized.parameters([0, 1, 2, -1, -2])
    def test_stack(self, axis):
        shape = (3, 5)
        n = 7
        values = [random_sparse(10, shape) for _ in range(n)]
        expected = tf.stack([tf.sparse.to_dense(v) for v in values], axis=axis)
        actual = tf.sparse.to_dense(ops.stack(values, axis=axis))
        self.assertAllEqual(actual, expected)

    @parameterized.parameters([0, 1, 2, -1, -2])
    def test_unstack(self, axis):
        shape = (3, 5, 7)
        st = random_sparse(50, shape)
        expected = tf.unstack(tf.sparse.to_dense(st), axis=axis)
        actual = [tf.sparse.to_dense(v) for v in ops.unstack(st, axis=axis)]
        self.assertAllEqual(actual, expected)

    @parameterized.parameters([0, 1, -1])
    def test_concat(self, axis):
        shape = (3, 5)
        n = 7
        values = [random_sparse(10, shape) for _ in range(n)]
        expected = tf.concat([tf.sparse.to_dense(v) for v in values], axis=axis)
        actual = tf.sparse.to_dense(ops.concat(values, axis=axis))
        self.assertAllEqual(actual, expected)

    @parameterized.parameters([([[1, 1], [2, 3], [5, 4]],)])
    def test_pad(self, padding):
        shape = (3, 5, 7)
        st = random_sparse(10, shape)
        actual = tf.sparse.to_dense(ops.pad(st, padding))
        expected = tf.pad(tf.sparse.to_dense(st), padding)
        self.assertAllClose(actual, expected)

    @parameterized.parameters([0, 1, -1])
    def test_gather(self, axis):
        shape = (50, 70)
        nnz = 2000
        st = random_sparse(nnz, shape)
        indices = tf.constant([2, 4], tf.int64)
        actual = ops.gather(st, indices, axis=axis)
        actual = tf.sparse.reorder(actual)
        actual = tf.sparse.to_dense(actual)
        expected = tf.gather(tf.sparse.to_dense(st), indices, axis=axis)
        self.assertAllEqual(actual, expected)

    def test_gather_custom(self):
        indices = tf.constant(
            [
                [0, 9],
                [0, 10],
                [0, 15],
                [0, 16],
                [1, 14],
                [1, 15],
                [1, 20],
                [1, 21],
                [2, 15],
                [2, 16],
                [2, 21],
                [2, 22],
                [3, 19],
                [3, 20],
                [3, 25],
                [3, 26],
                [4, 21],
                [4, 22],
                [4, 27],
                [4, 28],
                [5, 22],
                [5, 23],
                [5, 28],
                [5, 29],
            ],
            dtype=tf.int64,
        )
        dense_shape = tf.constant([6, 30], dtype=tf.int64)
        values = tf.range(1, 1 + tf.shape(indices, tf.int64)[0])
        st = tf.SparseTensor(indices, values, dense_shape)
        indices = tf.constant([9, 14, 15, 19, 21, 22], dtype=tf.int64)
        actual = ops.gather(st, indices, axis=1)
        actual = tf.sparse.to_dense(tf.sparse.reorder(actual))
        expected = tf.gather(tf.sparse.to_dense(st), indices, axis=1)
        self.assertAllEqual(actual, expected)

    def test_dense_hash_table_index_lookup(self):
        indices = tf.constant([0, 2, 5, 7, 10], tf.int64)
        query = tf.constant([5, 2, 2, 10], tf.int64)
        expected = tf.constant([2, 1, 1, 4], tf.int64)
        actual = ops.dense_hash_table_index_lookup(indices, query)
        self.assertAllEqual(actual, expected)

    def test_static_hash_table_index_lookup(self):
        indices = tf.constant([0, 2, 5, 7, 10], tf.int64)
        query = tf.constant([5, 2, 2, 10], tf.int64)
        expected = tf.constant([2, 1, 1, 4], tf.int64)
        actual = ops.static_hash_table_index_lookup(indices, query)
        self.assertAllEqual(actual, expected)

    def test_to_dense_index_lookup(self):
        indices = tf.constant([0, 2, 5, 7, 10], tf.int64)
        query = tf.constant([5, 2, 2, 10], tf.int64)
        expected = tf.constant([2, 1, 1, 4], tf.int64)
        actual = ops.to_dense_index_lookup(indices, query, 11)
        self.assertAllEqual(actual, expected)

    def test_sparse_tensor(self):
        values = tf.keras.Input(())
        indices = tf.keras.Input((2,), dtype=tf.int64)
        dense_shape = tf.keras.Input((), dtype=tf.int64)

        inputs = (indices, values, dense_shape)
        output = ops.sparse_tensor(*inputs)
        model = tf.keras.Model(inputs, output)

        st = random_sparse(5, (3, 7))
        actual = model((st.indices, st.values, st.dense_shape))

        self.assertAllEqual(actual.indices, st.indices)
        self.assertAllEqual(actual.values, st.values)
        self.assertAllEqual(actual.dense_shape, st.dense_shape)

    def test_sparse_tensor_necessary(self):
        with self.assertRaises(Exception):
            values = tf.keras.Input(())
            indices = tf.keras.Input((2,), dtype=tf.int64)
            dense_shape = tf.keras.Input((), dtype=tf.int64)

            inputs = (indices, values, dense_shape)
            tf.SparseTensor(*inputs)


if __name__ == "__main__":
    tf.test.main()
