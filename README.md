# [Sparse TensorFlow Utilities](https://github.com/jackd/stfu)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Contains:

- basic operations
  - pad
  - gather
  - boolean_mask
- keras layer for construction via components
- path to add `dense_shape` attribute to sparse `KerasTensor`s

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
