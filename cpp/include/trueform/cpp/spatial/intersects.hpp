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

#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/spatial/intersection_result.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Intersect two runtime primitives with scalar/broadcast semantics.
template <typename Real0, typename Real1, std::size_t Dims>
auto intersects(const primitive<Real0, Dims> &a,
                const primitive<Real1, Dims> &b) -> intersection_result;

/// @brief Intersect a mesh with one same-dimensional primitive or batch.
/// Query coordinates are converted to the form precision.
template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal>
auto intersects(const mesh<Index, FormReal, Dims, Ngon> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result;

/// @brief Intersect an edge mesh with one same-dimensional primitive or batch.
/// Query coordinates are converted to the form precision.
template <typename Index, typename FormReal, std::size_t Dims,
          typename PrimitiveReal>
auto intersects(const edge_mesh<Index, FormReal, Dims> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result;

/// @brief Intersect a point cloud with one same-dimensional primitive or batch.
/// Query coordinates are converted to the form precision.
template <typename FormReal, std::size_t Dims, typename PrimitiveReal>
auto intersects(const point_cloud<FormReal, Dims> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result;

/// @brief Symmetric primitive/form overloads.
template <typename PrimitiveReal, typename Index, typename FormReal,
          std::size_t Dims, std::size_t Ngon>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const mesh<Index, FormReal, Dims, Ngon> &form)
    -> intersection_result;
template <typename PrimitiveReal, typename Index, typename FormReal,
          std::size_t Dims>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const edge_mesh<Index, FormReal, Dims> &form)
    -> intersection_result;
template <typename PrimitiveReal, typename FormReal, std::size_t Dims>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const point_cloud<FormReal, Dims> &form) -> intersection_result;

/// @brief Intersect two same-dimensional forms.
/// Mixed form precision is rejected with `std::invalid_argument`; differing
/// dimensions have no native overload and are therefore a compile-time error.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result;
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const edge_mesh<Index1, Real1, Dims> &b) -> intersection_result;
template <typename Index0, typename Real0, typename Real1, std::size_t Dims,
          std::size_t Ngon0>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result;
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon1>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result;
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const edge_mesh<Index1, Real1, Dims> &b) -> intersection_result;
template <typename Index0, typename Real0, typename Real1, std::size_t Dims>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result;
template <typename Real0, typename Index1, typename Real1, std::size_t Dims,
          std::size_t Ngon1>
auto intersects(const point_cloud<Real0, Dims> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result;
template <typename Real0, typename Index1, typename Real1, std::size_t Dims>
auto intersects(const point_cloud<Real0, Dims> &a,
                const edge_mesh<Index1, Real1, Dims> &b) -> intersection_result;
template <typename Real0, typename Real1, std::size_t Dims>
auto intersects(const point_cloud<Real0, Dims> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result;

#define TF_CPP_DECLARE_INTERSECTS_PRIMITIVE(Real0, Real1, Dims)                \
  extern template auto intersects<Real0, Real1, Dims>(                         \
      const primitive<Real0, Dims> &, const primitive<Real1, Dims> &)          \
      -> intersection_result

TF_CPP_MATRIX_FOR_EACH_REAL_PAIR_DIMS(TF_CPP_DECLARE_INTERSECTS_PRIMITIVE)

#undef TF_CPP_DECLARE_INTERSECTS_PRIMITIVE

} // namespace tf::cpp
