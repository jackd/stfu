import tensorflow as tf

from stfu import layers
from stfu.test_utils import random_sparse


class OpsTest(tf.test.TestCase):
    def test_sparse_tensor(self):
        values = tf.keras.Input(())
        indices = tf.keras.Input((2,), dtype=tf.int64)
        dense_shape = tf.keras.Input((), dtype=tf.int64)

        inputs = (indices, values, dense_shape)
        output = layers.sparse_tensor(*inputs)
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
