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

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace tf::cpp {

/// @brief Scalar or batch result of a runtime distance query.
template <typename Real> class distance_result {
  std::variant<Real, nd_array<Real>> _value;

public:
  explicit distance_result(Real value) : _value(value) {}
  explicit distance_result(nd_array<Real> value) : _value(std::move(value)) {
    const auto &batch = std::get<nd_array<Real>>(_value);
    if (!batch.is_valid() || batch.ndim() != 1)
      throw std::invalid_argument(
          "distance_result: batch must be a valid one-dimensional array");
  }

  auto is_scalar() const -> bool {
    return std::holds_alternative<Real>(_value);
  }
  auto is_batch() const -> bool {
    return std::holds_alternative<nd_array<Real>>(_value);
  }
  auto scalar() const -> Real {
    if (!is_scalar())
      throw std::logic_error("distance_result: result is not scalar");
    return std::get<Real>(_value);
  }
  auto batch() const -> nd_array<Real> {
    if (!is_batch())
      throw std::logic_error("distance_result: result is not a batch");
    return std::get<nd_array<Real>>(_value).shallow_copy();
  }
};

/// @brief Compute squared distances between runtime primitives.
///
/// Two singles produce a scalar. A single and a batch use broadcast semantics,
/// while two batches are paired elementwise and must have equal lengths. Mixed
/// precision inputs are promoted to their common type. Both inputs must have
/// the same dimension.
template <typename Real0, typename Real1, std::size_t Dims>
auto distance2(const primitive<Real0, Dims> &a, const primitive<Real1, Dims> &b)
    -> distance_result<std::common_type_t<Real0, Real1>>;

/// @brief Compute distances between runtime primitives.
/// @copydetails distance2(const primitive<Real0, Dims> &,
///                        const primitive<Real1, Dims> &)
template <typename Real0, typename Real1, std::size_t Dims>
auto distance(const primitive<Real0, Dims> &a, const primitive<Real1, Dims> &b)
    -> distance_result<std::common_type_t<Real0, Real1>>;

/// @brief Compute squared distances from a mesh to a primitive or batch.
///
/// Query coordinates are converted to the form precision before computation.
/// Empty query batches return an empty batch without asking for the form's
/// tree. A scalar or nonempty query against a mesh with no faces throws
/// `std::invalid_argument`.
template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal>
auto distance2(const mesh<Index, FormReal, Dims, Ngon> &form,
               const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal>;

/// @brief Compute distances from a mesh to a primitive or batch.
template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal>
auto distance(const mesh<Index, FormReal, Dims, Ngon> &form,
              const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal>;

/// @brief Compute squared distances from a point cloud to a primitive or batch.
/// Empty query batches do not ask for the form's tree. A scalar or nonempty
/// query against a point cloud with no points throws `std::invalid_argument`.
template <typename FormReal, std::size_t Dims, typename PrimitiveReal>
auto distance2(const point_cloud<FormReal, Dims> &form,
               const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal>;

/// @brief Compute distances from a point cloud to a primitive or batch.
template <typename FormReal, std::size_t Dims, typename PrimitiveReal>
auto distance(const point_cloud<FormReal, Dims> &form,
              const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal>;

/// @brief Compute the squared distance between two meshes.
/// Mixed form precision and empty forms are rejected.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto distance2(const mesh<Index0, Real0, Dims, Ngon0> &a,
               const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the distance between two meshes.
/// Mixed form precision and empty forms are rejected.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto distance(const mesh<Index0, Real0, Dims, Ngon0> &a,
              const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the squared distance between a mesh and point cloud.
/// Mixed form precision and empty forms are rejected.
template <typename Index0, typename Real0, typename Real1, std::size_t Dims,
          std::size_t Ngon0>
auto distance2(const mesh<Index0, Real0, Dims, Ngon0> &a,
               const point_cloud<Real1, Dims> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the distance between a mesh and point cloud.
/// Mixed form precision and empty forms are rejected.
template <typename Index0, typename Real0, typename Real1, std::size_t Dims,
          std::size_t Ngon0>
auto distance(const mesh<Index0, Real0, Dims, Ngon0> &a,
              const point_cloud<Real1, Dims> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the squared distance between a point cloud and mesh.
/// Mixed form precision and empty forms are rejected.
template <typename Real0, typename Index1, typename Real1, std::size_t Dims,
          std::size_t Ngon1>
auto distance2(const point_cloud<Real0, Dims> &a,
               const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the distance between a point cloud and mesh.
/// Mixed form precision and empty forms are rejected.
template <typename Real0, typename Index1, typename Real1, std::size_t Dims,
          std::size_t Ngon1>
auto distance(const point_cloud<Real0, Dims> &a,
              const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the squared distance between two point clouds.
/// Mixed form precision and empty forms are rejected.
template <typename Real0, typename Real1, std::size_t Dims>
auto distance2(const point_cloud<Real0, Dims> &a,
               const point_cloud<Real1, Dims> &b)
    -> std::common_type_t<Real0, Real1>;

/// @brief Compute the distance between two point clouds.
/// Mixed form precision and empty forms are rejected.
template <typename Real0, typename Real1, std::size_t Dims>
auto distance(const point_cloud<Real0, Dims> &a,
              const point_cloud<Real1, Dims> &b)
    -> std::common_type_t<Real0, Real1>;

} // namespace tf::cpp
