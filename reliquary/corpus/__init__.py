"""Corpus generation on a frozen checkpoint.

Every decision in this package is a pure function over explicit arguments: a
weight-only node has to replay them and land on the same answer, so nothing
here may read a clock, an environment variable, or any global.
"""
