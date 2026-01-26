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

#pragma once

#include <initializer_list>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/curves_buffer.hpp>
#include <trueform/core/index_map.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/segments_buffer.hpp>
#include <trueform/core/transformation_like.hpp>
#include <trueform/core/unit_vectors_buffer.hpp>
#include <trueform/python/util/make_capsule.hpp>

namespace tf::py {

/**
 * Create a numpy array from raw pointer with proper ownership transfer
 * Handles empty arrays safely by using a shared dummy allocation
 * Shape is explicit, type T is inferred from the pointer
 */
template <typename Shape, typename T>
auto make_numpy_array(T *data, std::initializer_list<size_t> shape) {
  auto capsule = make_capsule<T>(data);
  return nanobind::ndarray<nanobind::numpy, T, Shape>(
      data ? data : reinterpret_cast<T *>(capsule.data()), shape, capsule);
}

/**
 * Create a numpy array from raw pointer with proper ownership transfer
 * Handles empty arrays safely by using a shared dummy allocation
 * Type T is inferred from the pointer, no shape constraint
 * Delegates to the shape-explicit version with shape<-1>
 */
template <typename T>
auto make_numpy_array(T *data, std::initializer_list<size_t> shape) {
  return make_numpy_array<nanobind::shape<-1>>(data, shape);
}

/**
 * Create a numpy array from tf::buffer by taking ownership
 * Extracts data pointer and releases ownership from the buffer
 * Shape is explicit, type T is inferred from the buffer
 */
template <typename Shape, typename T>
auto make_numpy_array(tf::buffer<T> &&buffer,
                      std::initializer_list<size_t> shape) {
  T *data = buffer.release();
  return make_numpy_array<Shape>(data, shape);
}

template <typename T> auto make_numpy_array(tf::buffer<T> &&buffer) {
  return make_numpy_array<nanobind::shape<-1>>(std::move(buffer),
                                               {buffer.size()});
}

template <typename T> auto make_numpy_array(tf::index_map_buffer<T> &&im) {
  return std::make_pair(make_numpy_array(std::move(im.f())),
                        make_numpy_array(std::move(im.kept_ids())));
}

/**
 * Create a numpy array from tf::buffer by taking ownership
 * Extracts data pointer and releases ownership from the buffer
 * Type T is inferred from the buffer, no shape constraint
 */
template <typename T>
auto make_numpy_array(tf::buffer<T> &&buffer,
                      std::initializer_list<size_t> shape) {
  return make_numpy_array<nanobind::shape<-1>>(std::move(buffer), shape);
}

/**
 * Create a numpy array from tf::blocked_buffer (faces) by taking ownership
 * Extracts (num_blocks, BlockSize) shaped array
 */
template <typename T, std::size_t BlockSize>
auto make_numpy_array(tf::blocked_buffer<T, BlockSize> &&blocked_buf) {
  auto num_blocks = blocked_buf.size();
  return make_numpy_array<nanobind::shape<-1, BlockSize>>(
      std::move(blocked_buf.data_buffer()),
      {static_cast<size_t>(num_blocks), BlockSize});
}

/**
 * Create a numpy array from tf::points_buffer by taking ownership
 * Extracts (num_points, Dims) shaped array
 */
template <typename RealT, std::size_t Dims>
auto make_numpy_array(tf::points_buffer<RealT, Dims> &&points_buf) {
  auto num_points = points_buf.size();
  return make_numpy_array<nanobind::shape<-1, Dims>>(
      std::move(points_buf.data_buffer()), {num_points, Dims});
}

template <typename RealT, std::size_t Dims>
auto make_numpy_array(
    tf::unit_vectors_buffer<RealT, Dims> &&unit_vectors_buffer) {
  auto num_vecs = unit_vectors_buffer.size();
  return make_numpy_array<nanobind::shape<-1, Dims>>(
      std::move(unit_vectors_buffer.data_buffer()), {num_vecs, Dims});
}

template <typename Index, typename RealT, std::size_t Dims>
auto make_numpy_array(tf::segments_buffer<Index, RealT, Dims> &&edge_mesh) {
  auto edges = make_numpy_array(std::move(edge_mesh.edges_buffer()));
  auto points = make_numpy_array(std::move(edge_mesh.points_buffer()));
  return std::make_pair(edges, points);
}

template <typename T, typename U>
auto make_numpy_array(tf::offset_block_buffer<T, U> &&ob_buff) {
  return std::make_pair(make_numpy_array(std::move(ob_buff.offsets_buffer())),
                        make_numpy_array(std::move(ob_buff.data_buffer())));
}

/**
 * Extract (faces, points) pair from tf::polygons_buffer by taking ownership
 * Returns std::pair of numpy arrays for faces and points
 */
template <typename Index, typename RealT, std::size_t NGon, std::size_t Dims>
auto make_numpy_array(tf::polygons_buffer<Index, RealT, NGon, Dims> &&mesh) {
  auto faces = make_numpy_array(std::move(mesh.faces_buffer()));
  auto points = make_numpy_array(std::move(mesh.points_buffer()));
  return std::make_pair(faces, points);
}
template <typename Index, typename RealT, std::size_t Dims>
auto make_numpy_array(tf::curves_buffer<Index, RealT, Dims> &&curves_buff) {
  return std::make_pair(
      make_numpy_array(std::move(curves_buff.paths_buffer())),
      make_numpy_array(std::move(curves_buff.points_buffer())));
}

/**
 * Create a numpy array from tf::transformation_like by copying
 * Extracts (Dims + 1, Dims + 1) shaped array (e.g., 4x4 for 3D)
 * The last row is filled with [0, 0, ..., 1]
 */
template <std::size_t Dims, typename Policy>
auto make_numpy_array(const tf::transformation_like<Dims, Policy> &trans) {
  using T = typename Policy::coordinate_type;
  constexpr std::size_t N = Dims + 1;
  T *data = new T[N * N];
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < N; ++j)
      data[i * N + j] = trans(i, j);
  // Last row: [0, 0, ..., 1]
  for (std::size_t j = 0; j < Dims; ++j)
    data[Dims * N + j] = T(0);
  data[Dims * N + Dims] = T(1);

  auto capsule = make_capsule<T>(data);
  return nanobind::ndarray<nanobind::numpy, T, nanobind::shape<N, N>>(
      data, {N, N}, capsule);
}

} // namespace tf::py
