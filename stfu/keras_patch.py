"""
Patch until PR is merged

https://github.com/keras-team/keras/pull/15199
"""
# pylint: disable=no-name-in-module
from tensorflow.python.keras.engine.keras_tensor import SparseKerasTensor
from tensorflow.python.keras.layers.core import _delegate_property

_delegate_property(SparseKerasTensor, "dense_shape")
