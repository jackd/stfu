import typing as tp

import tensorflow as tf


@tf.experimental.dispatch_for_api(tf.math.abs, {"x": tf.SparseTensor})
def _abs(x: tf.SparseTensor, name=None):
    with tf.name_scope(name or "abs"):
        return tf.sparse.map_values(tf.math.abs, x)


@tf.experimental.dispatch_for_api(tf.math.negative, {"x": tf.SparseTensor})
def _negative(x: tf.SparseTensor, name=None):
    with tf.name_scope(name or "negative"):
        return tf.sparse.map_values(tf.math.negative, x)


@tf.experimental.dispatch_for_api(tf.transpose, {"a": tf.SparseTensor})
def _transpose(a: tf.SparseTensor, perm=None, conjugate=False, name="transpose"):
    with tf.name_scope(name):
        if conjugate:
            a = tf.math.conj(a)
        return tf.sparse.transpose(a, perm=perm, name=name)


@tf.experimental.dispatch_for_api(tf.math.conj, {"x": tf.SparseTensor})
def _conj(x: tf.SparseTensor, name=None):
    with tf.name_scope(name or "conj"):
        return tf.sparse.map_values(tf.math.conj, x)


@tf.experimental.dispatch_for_api(
    tf.math.add, {"x": tf.SparseTensor, "y": tf.SparseTensor}
)
def _add_sparse_sparse(x: tf.SparseTensor, y: tf.SparseTensor, name=None):
    with tf.name_scope(name or "add_sparse_sparse"):
        return tf.sparse.add(x, y)


@tf.experimental.dispatch_for_api(tf.math.add, {"x": tf.SparseTensor, "y": tf.Tensor})
def _add_sparse_dense(x: tf.SparseTensor, y: tf.Tensor, name=None):
    with tf.name_scope(name or "add_sparse_dense"):
        dynamic_shape = tf.broadcast_dynamic_shape(
            tf.shape(y, x.dense_shape.dtype), x.dense_shape
        )
        # x = tf.broadcast_to(x, dynamic_shape)
        tf.debugging.assert_equal(
            x.dense_shape,
            dynamic_shape,
            "broadcast_to not implemented for `SparseTensor`s",
        )
        y = tf.broadcast_to(y, dynamic_shape)
        return tf.tensor_scatter_nd_add(y, x.indices, x.values)


@tf.experimental.dispatch_for_api(tf.math.add, {"x": tf.Tensor, "y": tf.SparseTensor})
def _add_dense_sparse(x: tf.Tensor, y: tf.SparseTensor, name=None):
    with tf.name_scope(name or "add_dense_sparse"):
        return _add_sparse_dense(y, x, name=None)


@tf.experimental.dispatch_for_api(
    tf.math.subtract, {"x": tf.SparseTensor}, {"y": tf.SparseTensor}
)
def _subtract(
    x: tp.Union[tf.SparseTensor, tf.Tensor],
    y: tp.Union[tf.SparseTensor, tf.Tensor],
    name=None,
):
    with tf.name_scope(name or "subtract"):
        return tf.math.add(x, tf.math.negative(y))


@tf.experimental.dispatch_for_api(
    tf.math.multiply, {"x": tf.SparseTensor, "y": tf.Tensor}
)
def _multiply_sparse_dense(x: tf.SparseTensor, y: tf.Tensor, name=None):
    with tf.name_scope(name or "multiply"):
        # TODO: the below is inefficient if sparse rank is greater than the dense rank
        dynamic_shape = tf.broadcast_dynamic_shape(
            tf.shape(y, x.dense_shape.dtype), x.dense_shape
        )
        # x = tf.broadcast_to(x, dynamic_shape)
        tf.debugging.assert_equal(
            x.dense_shape,
            dynamic_shape,
            "broadcast_to not supported for `SparseTensor`s",
        )
        y = tf.broadcast_to(y, dynamic_shape)
        if x.shape.ndims == 0:
            return tf.sparse.map_values(lambda x: x * y, x)
        return x.with_values(x.values * tf.gather_nd(y, x.indices))


@tf.experimental.dispatch_for_api(
    tf.math.multiply, {"x": tf.Tensor, "y": tf.SparseTensor}
)
def _multiply_dense_sparse(x: tf.Tensor, y: tf.SparseTensor, name=None):
    return _multiply_sparse_dense(y, x, name=name)


# @tf.experimental.dispatch_for_api(
#     tf.math.multiply, {"x": tf.SparseTensor, "y": tf.SparseTensor}
# )
# def _multiply_sparse_sparse(x: tf.SparseTensor, y: tf.SparseTensor, name=None):
#     with tf.name_scope(name or "multiply_sparse_sparse"):
#         tf.sets.


@tf.experimental.dispatch_for_api(
    tf.linalg.matmul, {"a": tf.SparseTensor, "b": tf.Tensor}
)
def _matmul(
    a: tf.SparseTensor,
    b: tf.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    a_is_sparse: bool = False,
    b_is_sparse: bool = False,
    output_type: tp.Optional[tf.DType] = None,
    name=None,
):
    del a_is_sparse, b_is_sparse
    if output_type is not None and output_type != a.dtype:
        raise NotImplementedError("output_type must be None or a.dtype")
    if transpose_a:
        if a.dtype.is_complex:
            raise NotImplementedError("transpose_a not supported for complex dtypes")
        adjoint_a = not adjoint_a
    if transpose_b:
        if b.dtype.is_complex:
            raise NotImplementedError("transpose_b not supported for complex dtypes")
        adjoint_b = not adjoint_b
    return tf.sparse.sparse_dense_matmul(
        a, b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name
    )


tf.SparseTensor.__abs__ = tf.math.abs
tf.SparseTensor.__neg__ = tf.math.negative
tf.SparseTensor.__add__ = tf.math.add
tf.SparseTensor.__sub__ = tf.math.subtract
tf.SparseTensor.__mul__ = tf.math.multiply
tf.SparseTensor.__matmul__ = tf.linalg.matmul
