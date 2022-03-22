import typing as tp

import numpy as np
import tensorflow as tf


def canonicalize_axis(axis: int, ndims: int) -> int:
    if axis < 0:
        axis += ndims
    assert 0 <= axis < ndims
    return axis


def squeeze(st: tf.SparseTensor, axis=None) -> tf.SparseTensor:
    """Same as tf.squeeze but supports `tf.SparseTensor` input."""
    if axis is None:
        axis = tuple(i for i in range(st.shape.ndims) if st.shape[i] == 1)
    else:
        if isinstance(axis, int):
            axis = (axis,)
        else:
            assert hasattr(axis, "__iter__"), axis
        axis = [canonicalize_axis(a, st.shape.ndims) for a in axis]
        assert all(isinstance(a, int) for a in axis), axis
        axis = np.unique(axis)  # remove duplicates / sort
        assert all(st.shape[a] == 1 or st.shape[a] is None for a in axis), (
            st.shape,
            axis,
        )
    remaining = list(range(st.shape.ndims))
    for a in axis[-1::-1]:
        del remaining[a]
    dense_shape = tf.gather(st.dense_shape, remaining, axis=0)
    indices = tf.gather(st.indices, remaining, axis=1)
    return sparse_tensor(indices, st.values, dense_shape)


def stack(values: tp.Sequence[tf.SparseTensor], axis: int = 0) -> tf.SparseTensor:
    """
    Same as `tf.stack` but for `tf.SparseTensor`s.

    If `axis == 0` and `values` are all in standard order, the output is also in
    standard ordering. Otherwise, consider using `tf.sparse.reorder` to reorder the
    output.
    """
    values = [tf.sparse.expand_dims(v, axis=axis) for v in values]
    return tf.sparse.concat(sp_inputs=values, axis=axis)


def unstack(
    value: tf.SparseTensor, num: tp.Optional[int] = None, axis: int = 0
) -> tp.List[tf.SparseTensor]:
    """Same as `tf.unstack` but for a `tf.SparseTensor`."""
    dense_shape = value.dense_shape
    axis = canonicalize_axis(axis, dense_shape.shape[0])
    if num is None:
        num = tf.get_static_value(dense_shape[axis])
        assert num is not None, (
            f"`dense_shape[{axis}]` must be statically known " "if `num` is not given"
        )
    else:
        tf.debugging.assert_equal(
            dense_shape[axis], tf.convert_to_tensor(num, dtype=dense_shape.dtype)
        )

    values = tf.sparse.split(value, num, axis=axis)
    return [squeeze(v, axis=axis) for v in values]


def concat(values: tp.Iterable[tf.SparseTensor], axis: int) -> tf.SparseTensor:
    """tf.concat but for `SparseTensor` values."""
    # no idea why `tf.sparse.concat` has reordered args compared to `tf.concat`
    return tf.sparse.concat(axis=axis, sp_inputs=values)


def pad_grid(
    indices: tf.Tensor, shape: tp.Union[tf.TensorShape, tf.Tensor], paddings
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Similar to `tf.pad` but applied to sparse indices/shape.

    Args:
        indices: [n, d] `d`-dimensional coordinates of `n` points.
        shape: [d] dense shape.
        paddings: [d, 2] leading/trailing padding for each dimension.

    Returns:
        padded_indices: indices + paddings[:, 0].
        padded_shape: shape + sum(paddings, axis=1).
    """
    if tf.is_tensor(paddings):
        paddings_static = tf.get_static_value(paddings)
    else:
        paddings_static = np.asarray(paddings)
        paddings = tf.convert_to_tensor(paddings, indices.dtype)
    if (
        paddings_static is not None
        and isinstance(shape, tf.TensorShape)
        and shape.is_fully_defined()
    ):
        shape = tf.TensorShape(np.asarray(shape) + np.sum(paddings_static, axis=1))
    else:
        shape = shape + tf.reduce_sum(paddings, axis=1)
    indices = indices + paddings[:, 0]
    return indices, shape


def pad(st: tf.SparseTensor, paddings) -> tf.SparseTensor:
    """Similar to `tf.pad` but accepts `SparseTensor` inputs."""
    shape = st.shape if st.shape.is_fully_defined() else st.dense_shape
    indices, shape = pad_grid(st.indices, shape, paddings)
    return sparse_tensor(indices, st.values, shape)


def indices_to_mask(
    indices: tf.Tensor, shape: tf.Tensor, dtype: tf.DType = tf.bool
) -> tf.Tensor:
    """
    Get a mask with true values at indices of the given shape.

    This can be used as an inverse to tf.where.

    Args:
        indices: [nnz, k] or [nnz] Tensor indices of True values.
        shape: [k] or [] (scalar) Tensor shape/size of output.

    Returns:
        Tensor of given shape and dtype.
    """
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    if indices.shape.ndims == 1:
        assert isinstance(shape, int) or shape.shape.ndims == 0
        indices = tf.expand_dims(indices, axis=1)
        if isinstance(shape, int):
            shape = tf.TensorShape([shape])
        else:
            shape = tf.expand_dims(shape, axis=0)
    else:
        indices.shape.assert_has_rank(2)
    assert indices.dtype.is_integer
    nnz = tf.shape(indices)[0]
    indices = tf.cast(indices, tf.int64)
    shape = tf.cast(shape, tf.int64)
    return tf.scatter_nd(indices, tf.ones((nnz,), dtype=dtype), shape)


def to_dense_index_lookup(
    indices: tf.Tensor, query: tf.Tensor, index_bound: tf.Tensor
) -> tf.Tensor:
    """
    Get the index into `indices` associated with `query` values.

    Implementation based on creating a dense vector of size `index_bound`. For large
    `index_bound` values, `dense_hash_table_index_lookup` may be faster.

    Args:
        indices: [ni] original indices in [0, index_bound).
        query: [nq], same dtype as `indices`. Every value of `query` should be in
            `indices`.
        index_bound: scalar bound on `indices`.

    Returns:
        [nq] index of `indices` for each entry in `query`. If `query` value is not in
            `indices`, the corresponding returned value is -1.
    """
    indices = tf.convert_to_tensor(indices, tf.int64)
    index_bound = tf.convert_to_tensor(index_bound, tf.int64)
    query = tf.convert_to_tensor(query, tf.int64)
    x = tf.scatter_nd(
        tf.expand_dims(indices, axis=-1),
        tf.range(1, 1 + tf.shape(indices, out_type=tf.int64)[0]),
        tf.expand_dims(index_bound, axis=-1),
    )
    return tf.gather(x, query, axis=0) - 1


def _dense_hash_table_index_lookup(args):
    indices, query = args
    values = tf.range(tf.size(indices, tf.int64), dtype=tf.int64)
    table = tf.lookup.experimental.DenseHashTable(tf.int64, tf.int64, -1, -1, -2)
    assert indices.dtype == tf.int64
    assert values.dtype == tf.int64
    assert query.dtype == tf.int64
    table.insert(indices, values)
    result = tf.reshape(table.lookup(tf.reshape(query, (-1,))), tf.shape(query))
    table.erase(indices)
    return result


def dense_hash_table_index_lookup(indices: tf.Tensor, query: tf.Tensor) -> tf.Tensor:
    """
    Get the index into `indices` associated with `query` values.

    Implementation based on `tf.lookup.experimental.DenseHashTable`. For small maximum
    `indices`, `to_dense_index_lookup` may be faster.

    Args:
        indices: [ni] original indices.
        query: [nq], same dtype as `indices`. Every value of `query` should be in
            `indices`.

    Returns:
        [nq] index of `indices` for each entry in `query`.
    """
    return tf.keras.layers.Lambda(_dense_hash_table_index_lookup)((indices, query))


def _mutable_hash_table_index_lookup(args):
    indices, query = args
    values = tf.range(tf.size(indices, tf.int64), dtype=tf.int64)
    table = tf.lookup.experimental.MutableHashTable(tf.int64, tf.int64, -1)
    assert indices.dtype == tf.int64
    assert values.dtype == tf.int64
    assert query.dtype == tf.int64
    table.insert(indices, values)
    result = tf.reshape(table.lookup(tf.reshape(query, (-1,))), tf.shape(query))
    table.remove(indices)
    return result


def mutable_hash_table_index_lookup(indices: tf.Tensor, query: tf.Tensor) -> tf.Tensor:
    """
    Get the index into `indices` associated with `query` values.

    Implementation based on `tf.lookup.experimental.DenseHashTable`. For small maximum
    `indices`, `to_dense_index_lookup` may be faster.

    Args:
        indices: [ni] original indices.
        query: [nq], same dtype as `indices`. Every value of `query` should be in
            `indices`.

    Returns:
        [nq] index of `indices` for each entry in `query`.
    """
    return tf.keras.layers.Lambda(_mutable_hash_table_index_lookup)((indices, query))


def static_hash_table_index_lookup(
    indices: tf.Tensor, query: tf.Tensor, default_value: tp.Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Get the index into `indices` associated with `query` values.

    Implementation based on `tf.lookup.experimental.StaticHashTable`. For small maximum
    `indices`, `to_dense_index_lookup` may be faster.

    Args:
        indices: [ni] original indices.
        query: [nq], same dtype as `indices`. Every value of `query` should be in
            `indices`.

    Returns:
        [nq] index of `indices` for each entry in `query`.
    """
    indices = tf.convert_to_tensor(indices)
    values = tf.range(tf.size(indices, indices.dtype), dtype=indices.dtype)
    init = tf.lookup.KeyValueTensorInitializer(indices, values)
    if default_value is None:
        default_value = -tf.ones((), indices.dtype)
    table = tf.lookup.StaticHashTable(init, default_value)
    result = tf.reshape(table.lookup(tf.reshape(query, (-1,))), tf.shape(query))
    return result


def _boolean_mask(
    a: tf.SparseTensor,
    mask: tf.Tensor,
    axis: int,
    gather_indices: tf.Tensor,
    out_size: tf.Tensor,
    return_values_mask: bool = False,
):
    """
    `SparseTensor` equivalent to tf.boolean_mask.

    Args:
        a: rank-k `tf.SparseTensor` with `nnz` non-zeros.
        mask: rank-1 bool Tensor.
        axis: int, axis on which to mask. Must be in [-k, k).
        gather_indices: pre-computed `tf.where(mask)`
        out_size: number of true entries in mask, tf.size(gather_indices).

    Returns:
        masked_a: SparseTensor masked along the given axis.
        values_mask: [nnz] bool Tensor indicating surviving non-zeros.
    """
    axis = canonicalize_axis(axis, a.shape.ndims)
    mask = tf.convert_to_tensor(mask, tf.bool)
    values_mask = tf.gather(mask, a.indices[:, axis], axis=0)
    dense_shape = tf.tensor_scatter_nd_update(a.dense_shape, [[axis]], [out_size])
    indices = tf.boolean_mask(a.indices, values_mask)
    indices = tf.unstack(indices, axis=-1)
    indices[axis] = mutable_hash_table_index_lookup(gather_indices, indices[axis])
    indices = tf.stack(indices, axis=-1)
    a = sparse_tensor(indices, tf.boolean_mask(a.values, values_mask), dense_shape)
    return (a, values_mask) if return_values_mask else a


def boolean_mask(
    a: tf.SparseTensor, mask: tf.Tensor, axis: int = 0, return_values_mask: bool = False
):
    """
    `SparseTensor` equivalent to `tf.boolean_mask`.

    Args:
        a: rank-k `tf.SparseTensor`.
        mask: rank-1 bool tensor.
        axis: axis along which to mask.
        return_values_mask: include `values_mask` in return

    Returns:
        If `return_values_mask`, returns `masked_a, values_mask`, otherwise just
            `masked_a`.
        masked_a: `tf.SparseTensor` masked along the given axis.
        values_mask: [nnz] bool tensor indicating surviving values.
    """
    i = tf.squeeze(tf.cast(tf.where(mask), tf.int64), axis=1)
    out_size = tf.math.count_nonzero(mask)
    return _boolean_mask(a, mask, axis=axis, gather_indices=i, out_size=out_size)


def gather(
    a: tf.SparseTensor,
    indices: tf.Tensor,
    axis: int = 0,
    return_values_mask: bool = False,
):
    """
    `tf.SparseTensor` equivalent to `tf.gather`.

    Assumes `indices` are sorted.

    Args:
        a: rank-k `tf.SparseTensor` with `nnz` non-zeros.
        indices: rank-1 int Tensor, rows or columns to keep.
        axis: int axis to apply gather to.
        return_values_mask: flag to return `values_mask`.

    Returns:
        If `return_values_mask`, returns `gathered_a, values_mask`, otherwise just
            `gathered_a`.
        gathered_a: `tf.SparseTensor` masked along the given axis.
        values_mask: [nnz] bool tensor indicating surviving values.
    """
    indices = tf.convert_to_tensor(indices, tf.int64)
    in_size = a.dense_shape[axis]
    out_size = tf.size(indices)
    mask = indices_to_mask(indices, in_size)
    return _boolean_mask(
        a,
        mask,
        axis=axis,
        gather_indices=indices,
        out_size=out_size,
        return_values_mask=return_values_mask,
    )


@tf.keras.utils.register_keras_serializable(package="stfu")
class SparseTensorLayer(tf.keras.layers.Layer):
    def call(self, args):
        return tf.SparseTensor(*args)


def sparse_tensor(
    indices: tf.Tensor, values: tf.Tensor, dense_shape: tf.Tensor
) -> tf.SparseTensor:
    """Create a SparseTensor, possibly via a keras layer."""
    if not tf.is_tensor(dense_shape):
        dense_shape = tf.convert_to_tensor(dense_shape, tf.int64)
    args = (indices, values, dense_shape)
    if any(tf.keras.backend.is_keras_tensor(t) for t in args):
        return SparseTensorLayer()(args)
    return tf.SparseTensor(*args)


def sparse_ones(indices, dense_shape, dtype: tf.DType = tf.float32) -> tf.SparseTensor:
    values = tf.ones((tf.shape(indices)[0],), dtype=dtype)
    return tf.SparseTensor(indices, values, dense_shape)


def diag(values: tf.Tensor) -> tf.SparseTensor:
    values.shape.assert_has_rank(1)
    size = tf.size(values, tf.int64)
    i = tf.range(size)
    return tf.SparseTensor(tf.tile(tf.expand_dims(i, 1), (1, 2)), values, (size, size))
