# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""FORMULA 14 - Entropy Regime Classifier (ERC) - Cython hot path.

Sample entropy is the only O(N^2) formula in the engine.  With N = 60 and
m = 2 that is ~1 800 distance comparisons per cycle, which is fast in numpy but
free in C.  This module is the compiled path; ``erc.py`` transparently falls
back to the numpy implementation when the extension has not been built.

Build:  python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, log

ctypedef cnp.float64_t DTYPE_t


def count_matches(cnp.ndarray[DTYPE_t, ndim=1] x, int m, double r):
    """Count template matches of length ``m`` with the Chebyshev metric.

    Returns the number of pairs (i < j) with
    ``max_k |x[i+k] - x[j+k]| < r``.
    """
    cdef Py_ssize_t n = x.shape[0] - m + 1
    cdef Py_ssize_t i, j, k
    cdef long count = 0
    cdef double d, best
    cdef double* data = <double*> cnp.PyArray_DATA(x)

    for i in range(n):
        for j in range(i + 1, n):
            best = 0.0
            for k in range(m):
                d = fabs(data[i + k] - data[j + k])
                if d > best:
                    best = d
                    if best >= r:
                        break
            if best < r:
                count += 1
    return count


def sample_entropy(cnp.ndarray[DTYPE_t, ndim=1] x, int m, double r, double fallback=2.5):
    """SampEn(m, r, N) = -ln(A / B)."""
    cdef long b = count_matches(x, m, r)
    cdef long a = count_matches(x, m + 1, r)
    if b == 0:
        return fallback
    if a == 0:
        # No length-(m+1) continuation for any match: maximally irregular.
        return fallback
    return -log(<double>a / <double>b)
