"""
Benchmark various lookup implementations.

Requires [tfbm](https://github.com/jackd/tfbm)

```bash
pip install git+https://github.com/jackd/tfbm.git
```
"""

import tensorflow as tf
from tfbm import Benchmark, benchmark

from stfu import ops

tf.random.set_seed(0)

max_val = int(1e8)
nnz = int(1e4)
query_size = 500

indices = tf.random.uniform((nnz,), maxval=max_val, dtype=tf.int64)
indices, _ = tf.unique(indices)
n = tf.size(indices, tf.int64)

query = tf.gather(indices, tf.random.uniform((query_size,), maxval=n, dtype=tf.int64))

indices = indices.numpy()
query = query.numpy()


class LookupBenchmark(Benchmark):

    BENCHMARK_SPEC = [
        benchmark(device="cpu"),
        benchmark(device="gpu"),
    ]

    @benchmark(args=(indices, max_val, query))
    def to_dense_index_lookup(self, indices, max_val, query):
        indices = tf.convert_to_tensor(indices, tf.int64)
        max_val = tf.convert_to_tensor(max_val, tf.int64)
        query = tf.convert_to_tensor(query, tf.int64)
        return ops.to_dense_index_lookup(indices, query, max_val)

    @benchmark(args=(indices, query))
    def dense_hash_table_index_lookup(self, indices, query):
        indices = tf.convert_to_tensor(indices, tf.int64)
        query = tf.convert_to_tensor(query, tf.int64)
        return ops.dense_hash_table_index_lookup(indices, query)

    @benchmark(args=(indices, query))
    def mutable_hash_table_index_lookup(self, indices, query):
        indices = tf.convert_to_tensor(indices, tf.int64)
        query = tf.convert_to_tensor(query, tf.int64)
        return ops.mutable_hash_table_index_lookup(indices, query)

    # @benchmark(args=(indices, max_val, query))
    # def sparse_has_table_lookup(self, indices, max_val, query):
    #     indices = tf.convert_to_tensor(indices, tf.int64)
    #     max_val = tf.convert_to_tensor(max_val, tf.int64)
    #     query = tf.convert_to_tensor(query, tf.int64)
    #     return ops.static_hash_table_index_lookup(indices, query)


if __name__ == "__main__":
    from tfbm.cli import main

    main()
