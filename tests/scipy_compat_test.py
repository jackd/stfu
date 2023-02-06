import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from stfu import scipy_compat, test_utils

del scipy_compat


class ScipyCompatTest(test_utils.SparseTestCase):
    def test_registered_conversion(self):
        n = 5
        actual = tf.convert_to_tensor(sp.eye(n, dtype=np.float32))
        expected = tf.eye(n)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
