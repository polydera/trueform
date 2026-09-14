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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Runtime kind of a primitive.
enum class primitive_kind {
  point,
  vector,
  segment,
  triangle,
  ray,
  line,
  plane,
  aabb,
  polygon
};

/// @brief Whether a runtime primitive stores one object or a batch.
enum class primitive_cardinality { single, batch };

/// @brief Validated runtime representation of a primitive or primitive batch.
///
/// Array storage is shared. Copies and views returned by data(), at() and
/// slice() retain the coordinate storage; use deep_copy() when storage must be
/// detached.
template <typename Real, std::size_t Dims = 3> class primitive {
  static_assert(Dims == 2 || Dims == 3, "primitive Dims must be 2 or 3");
  static_assert(matrix_carries_points_v<Real, Dims>,
                "this (real, dims) is not in the built C++ facade matrix; see "
                "TF_CPP_REALS / TF_CPP_DIMS");

  primitive_kind _kind;
  primitive_cardinality _cardinality;
  int _count;
  int _polygon_vertex_count;
  std::size_t _element_stride;
  nd_array<Real> _data;

public:
  primitive(primitive_kind kind, nd_array<Real> data);

  auto kind() const -> primitive_kind;
  auto cardinality() const -> primitive_cardinality;
  auto is_batch() const -> bool;
  auto count() const -> int;
  auto polygon_vertex_count() const -> int;
  auto element_stride() const -> std::size_t;

  auto data() const -> nd_array<Real>;
  auto at(int index) const -> primitive;
  auto slice(int start, int end) const -> primitive;
  auto shallow_copy() const -> primitive;
  auto deep_copy() const -> primitive;
};

#define TF_CPP_EXTERN_PRIMITIVE(Real, Dims)                                    \
  extern template class primitive<Real, Dims>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_PRIMITIVE)

#undef TF_CPP_EXTERN_PRIMITIVE

} // namespace tf::cpp
