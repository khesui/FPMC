"""
Argsort for 1D arrays
These codes come from
https://gist.github.com/hgrecco/818705ad1eef70f5f6ab

Thanks hgrecco @github
"""

import numpy as np
import numba as nb
from numba import jit

@jit(nopython=True)
def partition(values, idxs, left, right):
    """
    Partition method
    """

    piv = values[idxs[left]]
    i = left + 1
    j = right

    while True:
        while i <= j and values[idxs[i]] <= piv:
            i += 1
        while j >= i and values[idxs[j]] >= piv:
            j -= 1
        if j <= i:
            break

        idxs[i], idxs[j] = idxs[j], idxs[i]

    idxs[left], idxs[j] = idxs[j], idxs[left]

    return j


@jit(nopython=True)
def argsort1D(values):

    idxs = np.arange(values.shape[0])

    left = 0
    right = values.shape[0] - 1

    max_depth = np.int(right / 2)

    ndx = 0

    tmp = np.zeros((max_depth, 2), dtype=np.int64)

    tmp[ndx, 0] = left
    tmp[ndx, 1] = right

    ndx = 1
    while ndx > 0:

        ndx -= 1
        right = tmp[ndx, 1]
        left = tmp[ndx, 0]

        piv = partition(values, idxs, left, right)

        if piv - 1 > left:
            tmp[ndx, 0] = left
            tmp[ndx, 1] = piv - 1
            ndx += 1

        if piv + 1 < right:
            tmp[ndx, 0] = piv + 1
            tmp[ndx, 1] = right
            ndx += 1

    return idxs
