"""Portable NumPy 1.26.4 AVX512 argsort network, including equal-key ordering.

The original MTCNN final NMS uses NumPy's default quicksort. NumPy dispatches
to Intel's AVX512 key/index network on the reference host, and its equal-score
order affects which overlapping detection survives. This module emulates that
network with ordinary NumPy operations, without AVX512 instructions.

Ported from numpy/x86-simd-sort commit
0631a88763a4a0a4c9e84d5eeb0ec5d36053730b, bundled with NumPy v1.26.4:
src/avx512-64bit-argsort.hpp, src/avx512-64bit-keyvalue-networks.hpp,
src/avx512-common-argsort.h, src/avx512-common-qsort.h, and
src/avx512-64bit-common.h.
Copyright (C) 2022 Intel Corporation. SPDX-License-Identifier: BSD-3-Clause.
See argsort_compat.LICENSE.md for the original license.
"""
from __future__ import annotations

import math
import numpy as np


_ADJACENT = np.array([1, 0, 3, 2, 5, 4, 7, 6])
_REVERSE_FOUR = np.array([3, 2, 1, 0, 7, 6, 5, 4])
_REVERSE = np.array([7, 6, 5, 4, 3, 2, 1, 0])
_SWAP_TWO = np.array([2, 3, 0, 1, 6, 7, 4, 5])
_SWAP_FOUR = np.array([4, 5, 6, 7, 0, 1, 2, 3])
_MASKS = {mask: np.array([bool(mask & (1 << lane)) for lane in range(8)])
          for mask in (0xAA, 0xCC, 0xF0)}


def _cmp_merge(pair, permutation, mask):
    keys, indices = pair
    other_keys, other_indices = keys[permutation], indices[permutation]
    selected = np.where(_MASKS[mask], np.maximum(keys, other_keys), np.minimum(keys, other_keys))
    # AVX512 cmp_merge keeps the first input's index when its key compares equal.
    return selected, np.where(selected == keys, indices, other_indices)


def _sort_eight(pair):
    for permutation, mask in ((_ADJACENT, 0xAA), (_REVERSE_FOUR, 0xCC),
                              (_ADJACENT, 0xAA), (_REVERSE, 0xF0),
                              (_SWAP_TWO, 0xCC), (_ADJACENT, 0xAA)):
        pair = _cmp_merge(pair, permutation, mask)
    return pair


def _merge_eight(pair):
    for permutation, mask in ((_SWAP_FOUR, 0xF0), (_SWAP_TWO, 0xCC), (_ADJACENT, 0xAA)):
        pair = _cmp_merge(pair, permutation, mask)
    return pair


def _reverse(pair):
    return pair[0][_REVERSE], pair[1][_REVERSE]


def _coex(left, right):
    low = np.minimum(left[0], right[0])
    high = np.maximum(left[0], right[0])
    keep_left = low == left[0]
    return ((low, np.where(keep_left, left[1], right[1])),
            (high, np.where(keep_left, right[1], left[1])))


def _merge_blocks(blocks):
    count = len(blocks)
    half = count // 2
    low, high = [], []
    for i in range(half):
        small, large = _coex(blocks[i], _reverse(blocks[count - 1 - i]))
        low.append(small)
        high.append(_reverse(large))
    blocks = low + high[::-1]
    stride = half // 2
    while stride:
        for start in range(0, count, stride * 2):
            for i in range(start, start + stride):
                blocks[i], blocks[i + stride] = _coex(blocks[i], blocks[i + stride])
        stride //= 2
    return [_merge_eight(pair) for pair in blocks]


def _network(values, selected):
    count = len(selected)
    block_count = 1
    while block_count * 8 < count:
        block_count *= 2
    blocks = []
    for start in range(0, block_count * 8, 8):
        valid = max(0, min(8, count - start))
        keys = np.full(8, np.inf, dtype=values.dtype)
        indices = np.zeros(8, dtype=np.int64)
        indices[:valid] = selected[start:start + valid]
        keys[:valid] = values[indices[:valid]]
        blocks.append(_sort_eight((keys, indices)))
    size = 2
    while size <= block_count:
        for start in range(0, block_count, size):
            blocks[start:start + size] = _merge_blocks(blocks[start:start + size])
        size *= 2
    return np.concatenate([pair[1] for pair in blocks])[:count]


def _partition(values, indices, left, right, pivot):
    # Exact compress-store order from partition_avx512[_unrolled<vtype,4>].
    unroll = 4 if right - left > 256 else 1
    smallest, biggest = np.inf, -np.inf
    for _ in range((right - left) % (8 * unroll)):
        value = values[indices[left]]
        smallest, biggest = min(smallest, value), max(biggest, value)
        if value >= pivot:
            right -= 1
            indices[left], indices[right] = indices[right], indices[left]
        else:
            left += 1
    if left == right:
        return left, smallest, biggest
    l_store, r_store = left, right - 8

    def store(vector):
        nonlocal l_store, r_store, smallest, biggest
        keys = values[vector]
        high = keys >= pivot
        low_count = int((~high).sum())
        high_count = 8 - low_count
        indices[l_store:l_store + low_count] = vector[~high]
        indices[r_store + 8 - high_count:r_store + 8] = vector[high]
        smallest, biggest = min(smallest, keys.min()), max(biggest, keys.max())
        l_store += low_count
        r_store -= high_count

    if right - left == 8:
        store(indices[left:right].copy())
        return l_store, smallest, biggest
    first = [indices[left + 8 * i:left + 8 * (i + 1)].copy() for i in range(unroll)]
    last = [indices[right - 8 * (unroll - i):right - 8 * (unroll - i - 1)].copy()
            for i in range(unroll)]
    left += 8 * unroll
    right -= 8 * unroll
    while right != left:
        if r_store + 8 - right < left - l_store:
            right -= 8 * unroll
            start = right
        else:
            start = left
            left += 8 * unroll
        vectors = [indices[start + 8 * i:start + 8 * (i + 1)].copy() for i in range(unroll)]
        for vector in vectors:
            store(vector)
    for vector in first + last:
        store(vector)
    return l_store, smallest, biggest


def _quicksort(values, indices, left, right, iterations):
    if right < left:
        return
    if iterations <= 0:
        _std_sort(values, indices, left, right + 1)
        return
    if right + 1 - left <= 64:
        indices[left:right + 1] = _network(values, indices[left:right + 1])
        return
    step = (right - left) // 8
    sample = values[indices[left + step * np.arange(1, 9)]]
    pivot = np.sort(sample)[4]
    split, smallest, biggest = _partition(values, indices, left, right + 1, pivot)
    if pivot != smallest:
        _quicksort(values, indices, left, split - 1, iterations - 1)
    if pivot != biggest:
        _quicksort(values, indices, split, right, iterations - 1)


def _std_sort(values, indices, first, last):
    """Reference libstdc++ introsort: median-three partition plus insertion16.

    Intel's argsort invokes std::sort for its depth guard. Preserve that
    reference algorithm's equal-key behavior rather than calling another
    platform's default sorter.
    """
    def less(a, b):
        return values[indices[a]] < values[indices[b]]

    def swap(a, b):
        indices[a], indices[b] = indices[b], indices[a]

    def adjust_heap(begin, hole, length, value):
        top = hole
        child = hole
        while child < (length - 1) // 2:
            child = 2 * (child + 1)
            if values[indices[begin + child]] < values[indices[begin + child - 1]]:
                child -= 1
            indices[begin + hole] = indices[begin + child]
            hole = child
        if length % 2 == 0 and child == (length - 2) // 2:
            child = 2 * (child + 1)
            indices[begin + hole] = indices[begin + child - 1]
            hole = child - 1
        while hole > top:
            parent = (hole - 1) // 2
            if not values[indices[begin + parent]] < values[value]:
                break
            indices[begin + hole] = indices[begin + parent]
            hole = parent
        indices[begin + hole] = value

    def heap_sort(begin, end):
        length = end - begin
        if length < 2:
            return
        for parent in range((length - 2) // 2, -1, -1):
            adjust_heap(begin, parent, length, int(indices[begin + parent]))
        while end - begin > 1:
            end -= 1
            value = int(indices[end])
            indices[end] = indices[begin]
            adjust_heap(begin, 0, end - begin, value)

    def introsort(begin, end, depth):
        while end - begin > 16:
            if depth == 0:
                heap_sort(begin, end)
                return
            depth -= 1
            a, b, c = begin + 1, begin + (end - begin) // 2, end - 1
            if less(a, b):
                median = b if less(b, c) else c if less(a, c) else a
            else:
                median = a if less(a, c) else c if less(b, c) else b
            swap(begin, median)
            lo, hi = begin + 1, end
            while True:
                while less(lo, begin):
                    lo += 1
                hi -= 1
                while less(begin, hi):
                    hi -= 1
                if lo >= hi:
                    break
                swap(lo, hi)
                lo += 1
            introsort(lo, end, depth)
            end = lo

    if first == last:
        return
    introsort(first, last, 2 * int(math.log2(last - first)))
    # Stable insertion over the already partitioned range has the same result
    # as libstdc++'s insertion16 followed by unguarded linear insertions.
    for current in range(first + 1, last):
        value = int(indices[current])
        position = current
        while position > first and values[value] < values[indices[position - 1]]:
            indices[position] = indices[position - 1]
            position -= 1
        indices[position] = value


def argsort_reference(scores):
    """Emulate reference AVX512 float argsort without hardware-dependent ties."""
    values = np.asarray(scores)
    if values.ndim != 1 or values.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Reference NMS argsort expects a one-dimensional float array')
    if not np.isfinite(values).all():
        raise ValueError('Reference NMS argsort requires finite scores')
    indices = np.arange(len(values), dtype=np.int64)
    if len(values) > 1:
        _quicksort(values, indices, 0, len(values) - 1, 2 * int(math.log2(len(values))))
    return indices
