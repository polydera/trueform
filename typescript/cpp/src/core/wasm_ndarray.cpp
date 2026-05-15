/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/reductions.hpp"
#include "trueform/core/algorithm/parallel_iota.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <emscripten/bind.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

// ============================================================================
// stack: insert new axis
// stack([a[N,3], b[N,3]], axis=1) → [N,2,3]
// ============================================================================

template <typename T>
static auto stack(emscripten::val js_arrays, int axis)
    -> tf::ts::wasm_ndarray<T> {
  auto count = js_arrays["length"].as<int>();
  if (count == 0)
    throw std::runtime_error("stack: empty array list");

  std::vector<tf::ts::wasm_ndarray<T>> arrays;
  arrays.reserve(count);
  for (int i = 0; i < count; ++i)
    arrays.push_back(js_arrays[i].as<tf::ts::wasm_ndarray<T>>(
        emscripten::allow_raw_pointers()));

  auto &ref_shape = arrays[0].raw_shape();
  auto ndim = static_cast<int>(ref_shape.size());

  if (axis < 0)
    axis += ndim + 1;
  if (axis < 0 || axis > ndim)
    throw std::runtime_error("stack: axis out of range");

  for (int i = 1; i < count; ++i) {
    auto &s = arrays[i].raw_shape();
    if (static_cast<int>(s.size()) != ndim)
      throw std::runtime_error("stack: ndim mismatch");
    for (int d = 0; d < ndim; ++d)
      if (s[d] != ref_shape[d])
        throw std::runtime_error("stack: shape mismatch");
  }

  tf::small_vector<int, 3> out_shape;
  for (int d = 0; d < axis; ++d)
    out_shape.push_back(ref_shape[d]);
  out_shape.push_back(count);
  for (int d = axis; d < ndim; ++d)
    out_shape.push_back(ref_shape[d]);

  std::size_t outer = 1;
  for (int d = 0; d < axis; ++d)
    outer *= ref_shape[d];
  std::size_t inner = 1;
  for (int d = axis; d < ndim; ++d)
    inner *= ref_shape[d];

  auto total = outer * count * inner;
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (std::size_t o = 0; o < outer; ++o) {
    for (int n = 0; n < count; ++n) {
      auto *src = arrays[n].raw_data() + o * inner;
      std::memcpy(out + (o * count + n) * inner, src, inner * sizeof(T));
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// concatenate: join along existing axis
// concatenate([a[N,3], b[M,3]], axis=0) → [N+M,3]
// ============================================================================

template <typename T>
static auto concatenate(emscripten::val js_arrays, int axis)
    -> tf::ts::wasm_ndarray<T> {
  auto count = js_arrays["length"].as<int>();
  if (count == 0)
    throw std::runtime_error("concatenate: empty array list");

  std::vector<tf::ts::wasm_ndarray<T>> arrays;
  arrays.reserve(count);
  for (int i = 0; i < count; ++i)
    arrays.push_back(js_arrays[i].as<tf::ts::wasm_ndarray<T>>(
        emscripten::allow_raw_pointers()));

  auto &ref_shape = arrays[0].raw_shape();
  auto ndim = static_cast<int>(ref_shape.size());

  if (axis < 0)
    axis += ndim;
  if (axis < 0 || axis >= ndim)
    throw std::runtime_error("concatenate: axis out of range");

  for (int i = 1; i < count; ++i) {
    auto &s = arrays[i].raw_shape();
    if (static_cast<int>(s.size()) != ndim)
      throw std::runtime_error("concatenate: ndim mismatch");
    for (int d = 0; d < ndim; ++d)
      if (d != axis && s[d] != ref_shape[d])
        throw std::runtime_error("concatenate: shape mismatch on non-concat axis");
  }

  int concat_total = 0;
  for (int i = 0; i < count; ++i)
    concat_total += arrays[i].raw_shape()[axis];

  tf::small_vector<int, 3> out_shape = ref_shape;
  out_shape[axis] = concat_total;

  std::size_t outer = 1;
  for (int d = 0; d < axis; ++d)
    outer *= ref_shape[d];
  std::size_t inner = 1;
  for (int d = axis + 1; d < ndim; ++d)
    inner *= ref_shape[d];

  std::size_t total = 1;
  for (auto d : out_shape)
    total *= d;

  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (std::size_t o = 0; o < outer; ++o) {
    for (int i = 0; i < count; ++i) {
      auto axis_len = static_cast<std::size_t>(arrays[i].raw_shape()[axis]);
      auto block = axis_len * inner;
      auto *src = arrays[i].raw_data() + o * block;
      std::memcpy(out, src, block * sizeof(T));
      out += block;
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// tile: np.tile — repeat array along axes
// ============================================================================

template <typename T>
static auto tile(tf::ts::wasm_ndarray<T> &arr, emscripten::val js_reps)
    -> tf::ts::wasm_ndarray<T> {
  auto reps_len = js_reps["length"].as<int>();
  std::vector<int> reps(reps_len);
  for (int i = 0; i < reps_len; ++i)
    reps[i] = js_reps[i].as<int>();

  auto &in_shape = arr.raw_shape();
  auto in_ndim = static_cast<int>(in_shape.size());

  // Pad input shape / reps to same ndim (prepend 1s)
  auto out_ndim = std::max(in_ndim, reps_len);
  std::vector<int> padded_shape(out_ndim, 1);
  for (int i = 0; i < in_ndim; ++i)
    padded_shape[out_ndim - in_ndim + i] = in_shape[i];
  std::vector<int> padded_reps(out_ndim, 1);
  for (int i = 0; i < reps_len; ++i)
    padded_reps[out_ndim - reps_len + i] = reps[i];

  tf::small_vector<int, 3> out_shape;
  std::size_t total = 1;
  for (int d = 0; d < out_ndim; ++d) {
    auto s = padded_shape[d] * padded_reps[d];
    out_shape.push_back(s);
    total *= s;
  }

  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  std::memcpy(out, arr.raw_data(), arr.length() * sizeof(T));
  std::size_t current_len = arr.length();

  // Expand each axis from innermost to outermost
  for (int d = out_ndim - 1; d >= 0; --d) {
    auto rep = padded_reps[d];
    if (rep == 1)
      continue;

    std::size_t block = padded_shape[d];
    for (int k = d + 1; k < out_ndim; ++k)
      block *= out_shape[k];

    auto num_blocks = current_len / block;
    auto new_block = block * rep;

    // Expand backwards to avoid overwriting source
    for (std::size_t b = num_blocks; b-- > 0;) {
      auto *src_block = out + b * block;
      auto *dst_base = out + b * new_block;
      if (dst_base != src_block)
        std::memmove(dst_base, src_block, block * sizeof(T));
      for (int r = 1; r < rep; ++r)
        std::memcpy(dst_base + r * block, dst_base, block * sizeof(T));
    }

    current_len = num_blocks * new_block;
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// take: select slices along axis by int32 index array (np.take)
// ============================================================================

template <typename T>
static auto take(tf::ts::wasm_ndarray<T> &arr,
                 tf::ts::wasm_ndarray<int> &indices, int axis)
    -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto ndim = static_cast<int>(in_shape.size());
  if (axis < 0)
    axis += ndim;

  auto num_indices = static_cast<int>(indices.length());
  auto *idx_data = indices.raw_data();
  auto *src = arr.raw_data();

  auto [outer, axis_size, inner] =
      tf::ts::detail::axis_strides(in_shape, axis);

  tf::small_vector<int, 3> out_shape = in_shape;
  out_shape[axis] = num_indices;

  std::size_t total = static_cast<std::size_t>(outer) * num_indices * inner;
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (int o = 0; o < outer; ++o) {
    for (int i = 0; i < num_indices; ++i) {
      std::memcpy(
          out + (static_cast<std::size_t>(o) * num_indices + i) * inner,
          src + (static_cast<std::size_t>(o) * axis_size + idx_data[i]) * inner,
          inner * sizeof(T));
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// multi_take: per-axis Cartesian indexing
// specs is a JS array: each element is null (all), a number (single), or
// a wasm_ndarray<int> (fancy index). Unspecified trailing axes default to null.
// Output shape: null → keep dim, number → squeeze, array → use array length.
// ============================================================================

template <typename T>
static auto multi_take(tf::ts::wasm_ndarray<T> &arr, emscripten::val specs)
    -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto ndim = static_cast<int>(in_shape.size());
  auto n_specs = specs["length"].as<int>();

  // Per-axis: mode (0=all, 1=single, 2=array)
  tf::small_vector<int, 4> axis_mode(ndim);
  tf::small_vector<int, 4> axis_single(ndim);
  tf::small_vector<const int *, 4> axis_idx(ndim);
  tf::small_vector<int, 4> axis_count(ndim);
  tf::small_vector<tf::ts::wasm_ndarray<int>, 4> holders(ndim);

  for (int d = 0; d < ndim; ++d) {
    if (d >= n_specs || specs[d].isNull() || specs[d].isUndefined()) {
      axis_mode[d] = 0;
      axis_count[d] = in_shape[d];
      axis_idx[d] = nullptr;
    } else if (specs[d].isNumber()) {
      axis_mode[d] = 1;
      auto idx = specs[d].as<int>();
      axis_single[d] = idx < 0 ? idx + in_shape[d] : idx;
      axis_count[d] = 1;
      axis_idx[d] = nullptr;
    } else {
      axis_mode[d] = 2;
      holders[d] = specs[d].as<tf::ts::wasm_ndarray<int>>(
          emscripten::allow_raw_pointers());
      axis_idx[d] = holders[d].raw_data();
      axis_count[d] = static_cast<int>(holders[d].length());
    }
  }

  // Output shape (skip squeezed single-index axes)
  tf::small_vector<int, 3> out_shape;
  for (int d = 0; d < ndim; ++d)
    if (axis_mode[d] != 1)
      out_shape.push_back(axis_count[d]);
  if (out_shape.empty())
    out_shape.push_back(1);

  std::size_t total = 1;
  for (auto s : out_shape)
    total *= s;

  // Input strides
  tf::small_vector<std::size_t, 4> in_stride(ndim);
  in_stride[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d)
    in_stride[d] = in_stride[d + 1] * in_shape[d + 1];

  // Find contiguous trailing "all" axes for memcpy optimization
  int copy_from = ndim;
  for (int d = ndim - 1; d >= 0; --d) {
    if (axis_mode[d] == 0)
      copy_from = d;
    else
      break;
  }
  std::size_t inner = 1;
  for (int d = copy_from; d < ndim; ++d)
    inner *= in_shape[d];

  std::size_t n_outer = total / inner;

  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();
  auto *src = arr.raw_data();

  // Odometer over active (non-trailing-all) axes
  tf::small_vector<int, 4> coords(static_cast<std::size_t>(copy_from), 0);

  for (std::size_t i = 0; i < n_outer; ++i) {
    std::size_t src_off = 0;
    for (int d = 0; d < copy_from; ++d) {
      int si = axis_mode[d] == 0 ? coords[d]
             : axis_mode[d] == 1 ? axis_single[d]
             : axis_idx[d][coords[d]];
      src_off += si * in_stride[d];
    }

    std::memcpy(out + i * inner, src + src_off, inner * sizeof(T));

    for (int d = copy_from - 1; d >= 0; --d) {
      if (++coords[d] < axis_count[d])
        break;
      coords[d] = 0;
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// take_along_axis: per-element index along axis (np.take_along_axis)
// ============================================================================

template <typename T>
static auto take_along_axis(tf::ts::wasm_ndarray<T> &arr,
                            tf::ts::wasm_ndarray<int> &indices, int axis)
    -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto &idx_shape = indices.raw_shape();
  auto ndim = static_cast<int>(in_shape.size());
  if (axis < 0)
    axis += ndim;

  auto [outer, axis_size, inner] =
      tf::ts::detail::axis_strides(in_shape, axis);
  auto idx_axis_size = idx_shape[axis];

  auto *src = arr.raw_data();
  auto *idx_data = indices.raw_data();

  std::size_t total = indices.length();
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (int o = 0; o < outer; ++o) {
    for (int r = 0; r < idx_axis_size; ++r) {
      for (int i = 0; i < inner; ++i) {
        auto flat = (static_cast<std::size_t>(o) * idx_axis_size + r) * inner + i;
        auto idx = idx_data[flat];
        out[flat] = src[(static_cast<std::size_t>(o) * axis_size + idx) * inner + i];
      }
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              tf::small_vector<int, 3>(idx_shape));
}

// ============================================================================
// argsort: return int32 row permutation (lexicographic on flattened row)
// ============================================================================

template <typename T>
static auto argsort(tf::ts::wasm_ndarray<T> &arr)
    -> tf::ts::wasm_ndarray<int> {
  auto &in_shape = arr.raw_shape();
  auto *src = arr.raw_data();
  auto num_rows = in_shape[0];

  std::size_t stride = 1;
  for (int d = 1; d < static_cast<int>(in_shape.size()); ++d)
    stride *= in_shape[d];

  tf::buffer<int> buf;
  buf.allocate(num_rows);
  tf::parallel_iota(buf, 0);

  auto *out = buf.data();
  if (stride == 1) {
    tbb::parallel_sort(out, out + num_rows,
                       [src](int a, int b) { return src[a] < src[b]; });
  } else {
    tbb::parallel_sort(out, out + num_rows,
                       [src, stride](int a, int b) {
                         auto *ra = src + a * stride;
                         auto *rb = src + b * stride;
                         for (std::size_t k = 0; k < stride; ++k) {
                           if (ra[k] < rb[k])
                             return true;
                           if (ra[k] > rb[k])
                             return false;
                         }
                         return false;
                       });
  }

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(num_rows);
  return tf::ts::wasm_ndarray<int>::from_buffer(std::move(buf),
                                                std::move(out_shape));
}

// ============================================================================
// sort: return sorted copy (1D: direct sort, nD: argsort + take)
// ============================================================================

template <typename T>
static auto sort_copy(tf::ts::wasm_ndarray<T> &arr)
    -> tf::ts::wasm_ndarray<T> {
  auto len = arr.length();
  tf::buffer<T> buf;
  buf.allocate(len);
  std::memcpy(buf.data(), arr.raw_data(), len * sizeof(T));

  if (arr.raw_shape().size() == 1) {
    tbb::parallel_sort(buf.data(), buf.data() + len);
    return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), arr.raw_shape());
  }

  auto perm = argsort(arr);
  return take(arr, perm, 0);
}

// ============================================================================
// sort_inplace: sort in-place (1D: direct, nD: argsort + take + copy back)
// ============================================================================

template <typename T>
static void sort_inplace(tf::ts::wasm_ndarray<T> &arr) {
  if (arr.raw_shape().size() == 1) {
    tbb::parallel_sort(arr.raw_data(), arr.raw_data() + arr.length());
    return;
  }
  auto sorted = sort_copy(arr);
  std::memcpy(arr.raw_data(), sorted.raw_data(), arr.length() * sizeof(T));
}

// ============================================================================
// Row comparator helper for set operations on nD arrays
// ============================================================================

template <typename T>
static auto row_compare(const T *ra, const T *rb, std::size_t stride) -> int {
  for (std::size_t k = 0; k < stride; ++k) {
    if (ra[k] < rb[k])
      return -1;
    if (ra[k] > rb[k])
      return 1;
  }
  return 0;
}

// ============================================================================
// unique: remove duplicate rows from sorted array
// ============================================================================

template <typename T>
static auto unique(tf::ts::wasm_ndarray<T> &arr) -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto num_rows = in_shape[0];
  auto *src = arr.raw_data();

  if (num_rows == 0)
    return tf::ts::wasm_ndarray<T>::from_buffer(tf::buffer<T>{}, arr.raw_shape());

  std::size_t stride = 1;
  for (int d = 1; d < static_cast<int>(in_shape.size()); ++d)
    stride *= in_shape[d];

  // Count unique rows
  std::size_t count = 1;
  for (int i = 1; i < num_rows; ++i)
    if (row_compare(src + (i - 1) * stride, src + i * stride, stride) != 0)
      ++count;

  tf::buffer<T> buf;
  buf.allocate(count * stride);
  auto *out = buf.data();

  std::memcpy(out, src, stride * sizeof(T));
  out += stride;
  for (int i = 1; i < num_rows; ++i) {
    if (row_compare(src + (i - 1) * stride, src + i * stride, stride) != 0) {
      std::memcpy(out, src + i * stride, stride * sizeof(T));
      out += stride;
    }
  }

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(static_cast<int>(count));
  for (int d = 1; d < static_cast<int>(in_shape.size()); ++d)
    out_shape.push_back(in_shape[d]);

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// set_union: sorted union of two sorted arrays
// ============================================================================

template <typename T>
static auto set_union(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  auto &sa = a.raw_shape();
  auto &sb = b.raw_shape();
  auto na = sa[0];
  auto nb = sb[0];
  auto *da = a.raw_data();
  auto *db = b.raw_data();

  std::size_t stride = 1;
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    stride *= sa[d];

  tf::buffer<T> buf;
  buf.allocate((na + nb) * stride);
  auto *out = buf.data();
  std::size_t count = 0;

  int i = 0, j = 0;
  while (i < na && j < nb) {
    int cmp = row_compare(da + i * stride, db + j * stride, stride);
    if (cmp < 0) {
      std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
      ++i;
    } else if (cmp > 0) {
      std::memcpy(out + count * stride, db + j * stride, stride * sizeof(T));
      ++j;
    } else {
      std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
      ++i;
      ++j;
    }
    ++count;
  }
  while (i < na) {
    std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
    ++i;
    ++count;
  }
  while (j < nb) {
    std::memcpy(out + count * stride, db + j * stride, stride * sizeof(T));
    ++j;
    ++count;
  }

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(static_cast<int>(count));
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    out_shape.push_back(sa[d]);

  buf.reallocate(count * stride);
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// set_intersection: sorted intersection of two sorted arrays
// ============================================================================

template <typename T>
static auto set_intersection(tf::ts::wasm_ndarray<T> &a,
                             tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  auto &sa = a.raw_shape();
  auto &sb = b.raw_shape();
  auto na = sa[0];
  auto nb = sb[0];
  auto *da = a.raw_data();
  auto *db = b.raw_data();

  std::size_t stride = 1;
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    stride *= sa[d];

  auto max_out = std::min(na, nb);
  tf::buffer<T> buf;
  buf.allocate(max_out * stride);
  auto *out = buf.data();
  std::size_t count = 0;

  int i = 0, j = 0;
  while (i < na && j < nb) {
    int cmp = row_compare(da + i * stride, db + j * stride, stride);
    if (cmp < 0) {
      ++i;
    } else if (cmp > 0) {
      ++j;
    } else {
      std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
      ++count;
      ++i;
      ++j;
    }
  }

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(static_cast<int>(count));
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    out_shape.push_back(sa[d]);

  buf.reallocate(count * stride);
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// set_difference: elements in a not in b (both sorted)
// ============================================================================

template <typename T>
static auto set_difference(tf::ts::wasm_ndarray<T> &a,
                           tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  auto &sa = a.raw_shape();
  auto &sb = b.raw_shape();
  auto na = sa[0];
  auto nb = sb[0];
  auto *da = a.raw_data();
  auto *db = b.raw_data();

  std::size_t stride = 1;
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    stride *= sa[d];

  tf::buffer<T> buf;
  buf.allocate(na * stride);
  auto *out = buf.data();
  std::size_t count = 0;

  int i = 0, j = 0;
  while (i < na && j < nb) {
    int cmp = row_compare(da + i * stride, db + j * stride, stride);
    if (cmp < 0) {
      std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
      ++count;
      ++i;
    } else if (cmp > 0) {
      ++j;
    } else {
      ++i;
      ++j;
    }
  }
  while (i < na) {
    std::memcpy(out + count * stride, da + i * stride, stride * sizeof(T));
    ++count;
    ++i;
  }

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(static_cast<int>(count));
  for (int d = 1; d < static_cast<int>(sa.size()); ++d)
    out_shape.push_back(sa[d]);

  buf.reallocate(count * stride);
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// boolean_index: compact rows where mask is nonzero
// ============================================================================

template <typename T>
static auto boolean_index(tf::ts::wasm_ndarray<T> &arr,
                          tf::ts::wasm_ndarray<std::int8_t> &mask)
    -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto ndim = static_cast<int>(in_shape.size());
  auto *mask_data = mask.raw_data();
  auto *src = arr.raw_data();
  auto num_rows = in_shape[0];

  std::size_t row_stride = 1;
  for (int d = 1; d < ndim; ++d)
    row_stride *= in_shape[d];

  std::size_t count = 0;
  for (int i = 0; i < num_rows; ++i)
    if (mask_data[i])
      ++count;

  tf::small_vector<int, 3> out_shape;
  out_shape.push_back(static_cast<int>(count));
  for (int d = 1; d < ndim; ++d)
    out_shape.push_back(in_shape[d]);

  auto total = count * row_stride;
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (int i = 0; i < num_rows; ++i) {
    if (mask_data[i]) {
      std::memcpy(out, src + i * row_stride, row_stride * sizeof(T));
      out += row_stride;
    }
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// where: element-wise cond ? x : y
// ============================================================================

template <typename T>
static auto where(tf::ts::wasm_ndarray<std::int8_t> &cond,
                  tf::ts::wasm_ndarray<T> &x,
                  tf::ts::wasm_ndarray<T> &y) -> tf::ts::wasm_ndarray<T> {
  auto len = x.length();
  auto *c = cond.raw_data();
  auto *xd = x.raw_data();
  auto *yd = y.raw_data();

  tf::buffer<T> buf;
  buf.allocate(len);
  auto *out = buf.data();

  for (std::size_t i = 0; i < len; ++i)
    out[i] = c[i] ? xd[i] : yd[i];

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), x.raw_shape());
}

// ============================================================================
// transpose: permute axes (general) or swap last two (2D shorthand)
// ============================================================================

template <typename T>
static auto transpose(tf::ts::wasm_ndarray<T> &arr, emscripten::val js_axes)
    -> tf::ts::wasm_ndarray<T> {
  auto &in_shape = arr.raw_shape();
  auto ndim = static_cast<int>(in_shape.size());
  auto *src = arr.raw_data();

  std::vector<int> axes(ndim);
  if (js_axes.isUndefined() || js_axes.isNull()) {
    for (int i = 0; i < ndim; ++i)
      axes[i] = ndim - 1 - i;
  } else {
    for (int i = 0; i < ndim; ++i)
      axes[i] = js_axes[i].as<int>();
  }

  tf::small_vector<int, 3> out_shape;
  for (int i = 0; i < ndim; ++i)
    out_shape.push_back(in_shape[axes[i]]);

  std::vector<std::size_t> in_strides(ndim);
  in_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d)
    in_strides[d] = in_strides[d + 1] * in_shape[d + 1];

  std::vector<std::size_t> out_strides(ndim);
  out_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d)
    out_strides[d] = out_strides[d + 1] * out_shape[d + 1];

  auto total = arr.length();
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();

  for (std::size_t flat = 0; flat < total; ++flat) {
    std::size_t rem = flat;
    std::size_t src_idx = 0;
    for (int d = 0; d < ndim; ++d) {
      auto coord = rem / out_strides[d];
      rem %= out_strides[d];
      src_idx += coord * in_strides[axes[d]];
    }
    out[flat] = src[src_idx];
  }

  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf),
                                              std::move(out_shape));
}

// ============================================================================
// zeros / ones / full
// ============================================================================

template <typename T>
static auto zeros(emscripten::val js_shape) -> tf::ts::wasm_ndarray<T> {
  auto ndim = js_shape["length"].as<int>();
  tf::small_vector<int, 3> shape;
  std::size_t total = 1;
  for (int i = 0; i < ndim; ++i) {
    auto d = js_shape[i].as<int>();
    shape.push_back(d);
    total *= d;
  }
  tf::buffer<T> buf;
  buf.allocate(total);
  std::memset(buf.data(), 0, total * sizeof(T));
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), std::move(shape));
}

template <typename T>
static auto ones(emscripten::val js_shape) -> tf::ts::wasm_ndarray<T> {
  auto ndim = js_shape["length"].as<int>();
  tf::small_vector<int, 3> shape;
  std::size_t total = 1;
  for (int i = 0; i < ndim; ++i) {
    auto d = js_shape[i].as<int>();
    shape.push_back(d);
    total *= d;
  }
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();
  for (std::size_t i = 0; i < total; ++i)
    out[i] = T{1};
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), std::move(shape));
}

template <typename T>
static auto full(emscripten::val js_shape, T value) -> tf::ts::wasm_ndarray<T> {
  auto ndim = js_shape["length"].as<int>();
  tf::small_vector<int, 3> shape;
  std::size_t total = 1;
  for (int i = 0; i < ndim; ++i) {
    auto d = js_shape[i].as<int>();
    shape.push_back(d);
    total *= d;
  }
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();
  for (std::size_t i = 0; i < total; ++i)
    out[i] = value;
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), std::move(shape));
}

// ============================================================================
// arange / linspace
// ============================================================================

template <typename T>
static auto arange(T start, T stop, T step) -> tf::ts::wasm_ndarray<T> {
  std::size_t len = 0;
  if constexpr (std::is_floating_point_v<T>) {
    if (step > 0 && stop > start)
      len = static_cast<std::size_t>(
          std::ceil(static_cast<double>(stop - start) / static_cast<double>(step)));
    else if (step < 0 && stop < start)
      len = static_cast<std::size_t>(
          std::ceil(static_cast<double>(start - stop) / static_cast<double>(-step)));
  } else {
    if (step > 0 && stop > start)
      len = static_cast<std::size_t>((stop - start + step - 1) / step);
    else if (step < 0 && stop < start)
      len = static_cast<std::size_t>((start - stop - step - 1) / (-step));
  }

  tf::buffer<T> buf;
  buf.allocate(len);
  auto *out = buf.data();
  T val = start;
  for (std::size_t i = 0; i < len; ++i) {
    out[i] = val;
    val += step;
  }
  tf::small_vector<int, 3> shape;
  shape.push_back(static_cast<int>(len));
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), std::move(shape));
}

template <typename T>
static auto linspace(T start, T stop, int n) -> tf::ts::wasm_ndarray<T> {
  auto len = static_cast<std::size_t>(n);
  tf::buffer<T> buf;
  buf.allocate(len);
  auto *out = buf.data();
  if (n == 1) {
    out[0] = start;
  } else {
    auto step = (stop - start) / static_cast<T>(n - 1);
    for (std::size_t i = 0; i < len; ++i)
      out[i] = start + static_cast<T>(i) * step;
  }
  tf::small_vector<int, 3> shape;
  shape.push_back(n);
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), std::move(shape));
}

// ============================================================================
// clone: deep copy
// ============================================================================

template <typename T>
static auto clone(tf::ts::wasm_ndarray<T> &arr) -> tf::ts::wasm_ndarray<T> {
  auto len = arr.length();
  tf::buffer<T> buf;
  buf.allocate(len);
  std::memcpy(buf.data(), arr.raw_data(), len * sizeof(T));
  return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), arr.raw_shape());
}

// ============================================================================
// Async — sort / sort_inplace / argsort
// ============================================================================

template <typename T>
static auto async_sort(tf::ts::wasm_ndarray<T> &arr) -> tf::ts::promise_t {
  return tf::ts::promise([a = arr]() {
    return sort_copy(const_cast<tf::ts::wasm_ndarray<T> &>(a));
  });
}

template <typename T>
static auto async_sort_inplace(tf::ts::wasm_ndarray<T> &arr)
    -> tf::ts::promise_t {
  return tf::ts::promise([a = arr]() {
    sort_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}

template <typename T>
static auto async_argsort(tf::ts::wasm_ndarray<T> &arr) -> tf::ts::promise_t {
  return tf::ts::promise([a = arr]() {
    return argsort(const_cast<tf::ts::wasm_ndarray<T> &>(a));
  });
}

// ============================================================================
// Async — set operations
// ============================================================================

template <typename T>
static auto async_unique(tf::ts::wasm_ndarray<T> &arr) -> tf::ts::promise_t {
  return tf::ts::promise([a = arr]() {
    return unique(const_cast<tf::ts::wasm_ndarray<T> &>(a));
  });
}

template <typename T>
static auto async_set_union(tf::ts::wasm_ndarray<T> &a,
                            tf::ts::wasm_ndarray<T> &b) -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    return set_union(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                     const_cast<tf::ts::wasm_ndarray<T> &>(b));
  });
}

template <typename T>
static auto async_set_intersection(tf::ts::wasm_ndarray<T> &a,
                                   tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    return set_intersection(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                            const_cast<tf::ts::wasm_ndarray<T> &>(b));
  });
}

template <typename T>
static auto async_set_difference(tf::ts::wasm_ndarray<T> &a,
                                 tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    return set_difference(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                          const_cast<tf::ts::wasm_ndarray<T> &>(b));
  });
}

EMSCRIPTEN_BINDINGS(trueform_buffers) {
  emscripten::class_<tf::ts::wasm_ndarray<int>>("NativeInt32NDArray")
      .class_function("from_js", &tf::ts::wasm_ndarray<int>::from_js)
      .function("data", &tf::ts::wasm_ndarray<int>::data)
      .function("size", &tf::ts::wasm_ndarray<int>::size)
      .function("ndim", &tf::ts::wasm_ndarray<int>::ndim)
      .function("shape", &tf::ts::wasm_ndarray<int>::shape)
      .function("shape_at", &tf::ts::wasm_ndarray<int>::shape_at)
      .function("set_shape", &tf::ts::wasm_ndarray<int>::set_shape)
      .function("row", &tf::ts::wasm_ndarray<int>::row)
      .function("slice", &tf::ts::wasm_ndarray<int>::slice)
      .function("destroy", &tf::ts::wasm_ndarray<int>::destroy)
      .function("is_valid", &tf::ts::wasm_ndarray<int>::is_valid)
      .function("shallow_copy", &tf::ts::wasm_ndarray<int>::shallow_copy);

  emscripten::class_<tf::ts::wasm_ndarray<std::int8_t>>("NativeInt8NDArray")
      .class_function("from_js", &tf::ts::wasm_ndarray<std::int8_t>::from_js)
      .function("data", &tf::ts::wasm_ndarray<std::int8_t>::data)
      .function("size", &tf::ts::wasm_ndarray<std::int8_t>::size)
      .function("ndim", &tf::ts::wasm_ndarray<std::int8_t>::ndim)
      .function("shape", &tf::ts::wasm_ndarray<std::int8_t>::shape)
      .function("shape_at", &tf::ts::wasm_ndarray<std::int8_t>::shape_at)
      .function("set_shape", &tf::ts::wasm_ndarray<std::int8_t>::set_shape)
      .function("row", &tf::ts::wasm_ndarray<std::int8_t>::row)
      .function("slice", &tf::ts::wasm_ndarray<std::int8_t>::slice)
      .function("destroy", &tf::ts::wasm_ndarray<std::int8_t>::destroy)
      .function("is_valid", &tf::ts::wasm_ndarray<std::int8_t>::is_valid)
      .function("shallow_copy", &tf::ts::wasm_ndarray<std::int8_t>::shallow_copy);

  emscripten::class_<tf::ts::wasm_ndarray<float>>("NativeFloat32NDArray")
      .class_function("from_js", &tf::ts::wasm_ndarray<float>::from_js)
      .function("data", &tf::ts::wasm_ndarray<float>::data)
      .function("size", &tf::ts::wasm_ndarray<float>::size)
      .function("ndim", &tf::ts::wasm_ndarray<float>::ndim)
      .function("shape", &tf::ts::wasm_ndarray<float>::shape)
      .function("shape_at", &tf::ts::wasm_ndarray<float>::shape_at)
      .function("set_shape", &tf::ts::wasm_ndarray<float>::set_shape)
      .function("row", &tf::ts::wasm_ndarray<float>::row)
      .function("slice", &tf::ts::wasm_ndarray<float>::slice)
      .function("destroy", &tf::ts::wasm_ndarray<float>::destroy)
      .function("is_valid", &tf::ts::wasm_ndarray<float>::is_valid)
      .function("shallow_copy", &tf::ts::wasm_ndarray<float>::shallow_copy);

  emscripten::class_<tf::ts::wasm_ndarray<double>>("NativeFloat64NDArray")
      .class_function("from_js", &tf::ts::wasm_ndarray<double>::from_js)
      .function("data", &tf::ts::wasm_ndarray<double>::data)
      .function("size", &tf::ts::wasm_ndarray<double>::size)
      .function("ndim", &tf::ts::wasm_ndarray<double>::ndim)
      .function("shape", &tf::ts::wasm_ndarray<double>::shape)
      .function("shape_at", &tf::ts::wasm_ndarray<double>::shape_at)
      .function("set_shape", &tf::ts::wasm_ndarray<double>::set_shape)
      .function("row", &tf::ts::wasm_ndarray<double>::row)
      .function("slice", &tf::ts::wasm_ndarray<double>::slice)
      .function("destroy", &tf::ts::wasm_ndarray<double>::destroy)
      .function("is_valid", &tf::ts::wasm_ndarray<double>::is_valid)
      .function("shallow_copy", &tf::ts::wasm_ndarray<double>::shallow_copy);

  // --- stack ---
  emscripten::function("stack_float32", &stack<float>);
  emscripten::function("stack_float64", &stack<double>);
  emscripten::function("stack_int32", &stack<int>);
  emscripten::function("stack_int8", &stack<std::int8_t>);

  // --- concatenate ---
  emscripten::function("concatenate_float32", &concatenate<float>);
  emscripten::function("concatenate_float64", &concatenate<double>);
  emscripten::function("concatenate_int32", &concatenate<int>);
  emscripten::function("concatenate_int8", &concatenate<std::int8_t>);

  // --- tile ---
  emscripten::function("tile_float32", &tile<float>);
  emscripten::function("tile_float64", &tile<double>);
  emscripten::function("tile_int32", &tile<int>);
  emscripten::function("tile_int8", &tile<std::int8_t>);

  // --- take ---
  emscripten::function("take_float32", &take<float>);
  emscripten::function("take_float64", &take<double>);
  emscripten::function("take_int32", &take<int>);
  emscripten::function("take_int8", &take<std::int8_t>);

  // --- multi_take ---
  emscripten::function("multi_take_float32", &multi_take<float>);
  emscripten::function("multi_take_float64", &multi_take<double>);
  emscripten::function("multi_take_int32", &multi_take<int>);
  emscripten::function("multi_take_int8", &multi_take<std::int8_t>);

  // --- take_along_axis ---
  emscripten::function("take_along_axis_float32", &take_along_axis<float>);
  emscripten::function("take_along_axis_float64", &take_along_axis<double>);
  emscripten::function("take_along_axis_int32", &take_along_axis<int>);
  emscripten::function("take_along_axis_int8", &take_along_axis<std::int8_t>);

  // --- argsort ---
  emscripten::function("argsort_float32", &argsort<float>);
  emscripten::function("argsort_float64", &argsort<double>);
  emscripten::function("argsort_int32", &argsort<int>);
  emscripten::function("argsort_int8", &argsort<std::int8_t>);

  // --- sort ---
  emscripten::function("sort_float32", &sort_copy<float>);
  emscripten::function("sort_float64", &sort_copy<double>);
  emscripten::function("sort_int32", &sort_copy<int>);
  emscripten::function("sort_int8", &sort_copy<std::int8_t>);

  // --- sort_inplace ---
  emscripten::function("sort_inplace_float32", &sort_inplace<float>);
  emscripten::function("sort_inplace_float64", &sort_inplace<double>);
  emscripten::function("sort_inplace_int32", &sort_inplace<int>);
  emscripten::function("sort_inplace_int8", &sort_inplace<std::int8_t>);

  // --- async sort ---
  emscripten::function("dispatch_sort_float32", &async_sort<float>);
  emscripten::function("dispatch_sort_float64", &async_sort<double>);
  emscripten::function("dispatch_sort_int32", &async_sort<int>);
  emscripten::function("dispatch_sort_int8", &async_sort<std::int8_t>);

  // --- async sort_inplace ---
  emscripten::function("dispatch_sort_inplace_float32", &async_sort_inplace<float>);
  emscripten::function("dispatch_sort_inplace_float64", &async_sort_inplace<double>);
  emscripten::function("dispatch_sort_inplace_int32", &async_sort_inplace<int>);
  emscripten::function("dispatch_sort_inplace_int8", &async_sort_inplace<std::int8_t>);

  // --- async argsort ---
  emscripten::function("dispatch_argsort_float32", &async_argsort<float>);
  emscripten::function("dispatch_argsort_float64", &async_argsort<double>);
  emscripten::function("dispatch_argsort_int32", &async_argsort<int>);
  emscripten::function("dispatch_argsort_int8", &async_argsort<std::int8_t>);

  // --- unique ---
  emscripten::function("unique_float32", &unique<float>);
  emscripten::function("unique_float64", &unique<double>);
  emscripten::function("unique_int32", &unique<int>);
  emscripten::function("unique_int8", &unique<std::int8_t>);
  emscripten::function("dispatch_unique_float32", &async_unique<float>);
  emscripten::function("dispatch_unique_float64", &async_unique<double>);
  emscripten::function("dispatch_unique_int32", &async_unique<int>);
  emscripten::function("dispatch_unique_int8", &async_unique<std::int8_t>);

  // --- set_union ---
  emscripten::function("set_union_float32", &set_union<float>);
  emscripten::function("set_union_float64", &set_union<double>);
  emscripten::function("set_union_int32", &set_union<int>);
  emscripten::function("set_union_int8", &set_union<std::int8_t>);
  emscripten::function("dispatch_set_union_float32", &async_set_union<float>);
  emscripten::function("dispatch_set_union_float64", &async_set_union<double>);
  emscripten::function("dispatch_set_union_int32", &async_set_union<int>);
  emscripten::function("dispatch_set_union_int8", &async_set_union<std::int8_t>);

  // --- set_intersection ---
  emscripten::function("set_intersection_float32", &set_intersection<float>);
  emscripten::function("set_intersection_float64", &set_intersection<double>);
  emscripten::function("set_intersection_int32", &set_intersection<int>);
  emscripten::function("set_intersection_int8", &set_intersection<std::int8_t>);
  emscripten::function("dispatch_set_intersection_float32", &async_set_intersection<float>);
  emscripten::function("dispatch_set_intersection_float64", &async_set_intersection<double>);
  emscripten::function("dispatch_set_intersection_int32", &async_set_intersection<int>);
  emscripten::function("dispatch_set_intersection_int8", &async_set_intersection<std::int8_t>);

  // --- set_difference ---
  emscripten::function("set_difference_float32", &set_difference<float>);
  emscripten::function("set_difference_float64", &set_difference<double>);
  emscripten::function("set_difference_int32", &set_difference<int>);
  emscripten::function("set_difference_int8", &set_difference<std::int8_t>);
  emscripten::function("dispatch_set_difference_float32", &async_set_difference<float>);
  emscripten::function("dispatch_set_difference_float64", &async_set_difference<double>);
  emscripten::function("dispatch_set_difference_int32", &async_set_difference<int>);
  emscripten::function("dispatch_set_difference_int8", &async_set_difference<std::int8_t>);

  // --- boolean_index ---
  emscripten::function("boolean_index_float32", &boolean_index<float>);
  emscripten::function("boolean_index_float64", &boolean_index<double>);
  emscripten::function("boolean_index_int32", &boolean_index<int>);
  emscripten::function("boolean_index_int8", &boolean_index<std::int8_t>);

  // --- where ---
  emscripten::function("where_float32", &where<float>);
  emscripten::function("where_float64", &where<double>);
  emscripten::function("where_int32", &where<int>);
  emscripten::function("where_int8", &where<std::int8_t>);

  // --- transpose ---
  emscripten::function("transpose_float32", &transpose<float>);
  emscripten::function("transpose_float64", &transpose<double>);
  emscripten::function("transpose_int32", &transpose<int>);
  emscripten::function("transpose_int8", &transpose<std::int8_t>);

  // --- zeros ---
  emscripten::function("zeros_float32", &zeros<float>);
  emscripten::function("zeros_float64", &zeros<double>);
  emscripten::function("zeros_int32", &zeros<int>);
  emscripten::function("zeros_int8", &zeros<std::int8_t>);

  // --- ones ---
  emscripten::function("ones_float32", &ones<float>);
  emscripten::function("ones_float64", &ones<double>);
  emscripten::function("ones_int32", &ones<int>);
  emscripten::function("ones_int8", &ones<std::int8_t>);

  // --- full ---
  emscripten::function("full_float32", &full<float>);
  emscripten::function("full_float64", &full<double>);
  emscripten::function("full_int32", &full<int>);
  emscripten::function("full_int8", &full<std::int8_t>);

  // --- arange ---
  emscripten::function("arange_float32", &arange<float>);
  emscripten::function("arange_float64", &arange<double>);
  emscripten::function("arange_int32", &arange<int>);

  // --- linspace ---
  emscripten::function("linspace_float32", &linspace<float>);
  emscripten::function("linspace_float64", &linspace<double>);

  // --- clone ---
  emscripten::function("clone_float32", &clone<float>);
  emscripten::function("clone_float64", &clone<double>);
  emscripten::function("clone_int32", &clone<int>);
  emscripten::function("clone_int8", &clone<std::int8_t>);
}
