import tensorflow as tf

import stfu.keras_patch  # pylint: disable=unused-import


class KerasSparseTensorTest(tf.test.TestCase):
    def test_dense_shape_property(self):
        st = tf.keras.Input((5,), sparse=True)
        st.dense_shape


if __name__ == "__main__":
    tf.test.main()
