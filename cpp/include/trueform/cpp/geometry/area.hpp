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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tf::cpp {

/// @brief Scalar or batch result of a runtime primitive area query.
template <typename Real> class area_result {
  std::variant<Real, nd_array<Real>> _value;

public:
  explicit area_result(Real value) : _value(value) {}
  explicit area_result(nd_array<Real> value) : _value(std::move(value)) {
    const auto &batch = std::get<nd_array<Real>>(_value);
    if (!batch.is_valid() || batch.ndim() != 1)
      throw std::invalid_argument(
          "area_result: batch must be a valid one-dimensional array");
  }

  auto is_scalar() const -> bool {
    return std::holds_alternative<Real>(_value);
  }
  auto is_batch() const -> bool {
    return std::holds_alternative<nd_array<Real>>(_value);
  }
  auto scalar() const -> Real {
    if (!is_scalar())
      throw std::logic_error("area_result: result is not scalar");
    return std::get<Real>(_value);
  }
  auto batch() const -> nd_array<Real> {
    if (!is_batch())
      throw std::logic_error("area_result: result is not a batch");
    return std::get<nd_array<Real>>(_value).shallow_copy();
  }
};

/// @brief Compute area for a runtime triangle or polygon.
///
/// A single primitive produces a scalar result. A batch produces one area per
/// primitive, including a valid empty array for an empty batch.
template <typename Real, std::size_t Dims>
auto area(const primitive<Real, Dims> &value) -> area_result<Real>;

/// @brief Compute the total surface area of a mesh in its transformed frame.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto area(const mesh<Index, Real, Dims, Ngon> &value) -> Real;

#define TF_CPP_EXTERN_PRIMITIVE_AREA(Real, Dims)                               \
  extern template auto area<Real, Dims>(const primitive<Real, Dims> &)         \
      -> area_result<Real>

#define TF_CPP_EXTERN_MESH_AREA(Index, Real, Dims, Ngon)                       \
  extern template auto area<Index, Real, Dims, Ngon>(                          \
      const mesh<Index, Real, Dims, Ngon> &) -> Real

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_PRIMITIVE_AREA)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_MESH_AREA)

#undef TF_CPP_EXTERN_MESH_AREA
#undef TF_CPP_EXTERN_PRIMITIVE_AREA

} // namespace tf::cpp
