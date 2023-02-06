import scipy.sparse as sp
import tensorflow as tf


def sp_to_tf(
    matrix: sp.spmatrix, dtype=None, name=None, as_ref: bool = False
) -> tf.SparseTensor:
    if as_ref:
        raise NotImplementedError("as_ref not implemented")
    matrix: sp.coo_matrix = matrix.tocoo()
    with tf.name_scope(name or "sp_to_tf"):
        values = tf.convert_to_tensor(matrix.data, dtype)
        indices = tf.stack((matrix.row, matrix.col), axis=-1)
        indices = tf.cast(indices, tf.int64)
        return tf.SparseTensor(indices, values, matrix.shape)


tf.register_tensor_conversion_function(sp.spmatrix, sp_to_tf)
