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
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

namespace tf::cpp {

/// @brief Parametric bounds for ray casting.
///
/// Each valid per-result array replaces its corresponding scalar bound. A
/// bound array must have shape [N], where N is the broadcast result count.
/// Bounds are inclusive. Inverted bounds are permitted and produce misses.
template <typename Real> struct ray_cast_options {
  Real min_t = Real{0};
  Real max_t = std::numeric_limits<Real>::max();
  nd_array<Real> min_ts;
  nd_array<Real> max_ts;
};

/// @brief Result for a scalar ray cast.
template <typename Index, typename Real> struct ray_cast_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "ray_cast_result requires an unqualified supported index type");

  bool hit = false;
  Real t = Real{0};
  Index element_id = Index{-1};
};

/// @brief Result for a broadcast or pairwise primitive ray cast.
template <typename Real> struct ray_cast_primitive_batch_result {
  nd_array<std::int8_t> hits;
  nd_array<Real> ts;
};

/// @brief Result for a batch ray cast against a mesh or point cloud.
template <typename Index, typename Real> struct ray_cast_form_batch_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "ray_cast_form_batch_result requires an unqualified supported "
                "index type");

  nd_array<std::int8_t> hits;
  nd_array<Real> ts;
  nd_array<Index> element_ids;
};

/// @brief Scalar-or-batch result of casting runtime rays against primitives.
template <typename Real> class ray_cast_primitive_result {
  std::variant<ray_cast_result<default_index_t, Real>,
               ray_cast_primitive_batch_result<Real>>
      _value;

public:
  ray_cast_primitive_result(ray_cast_result<default_index_t, Real> value)
      : _value(std::move(value)) {}
  ray_cast_primitive_result(ray_cast_primitive_batch_result<Real> value)
      : _value(std::move(value)) {}

  auto is_scalar() const -> bool {
    return std::holds_alternative<ray_cast_result<default_index_t, Real>>(
        _value);
  }
  auto is_batch() const -> bool { return !is_scalar(); }
  auto scalar() -> ray_cast_result<default_index_t, Real> & {
    return std::get<ray_cast_result<default_index_t, Real>>(_value);
  }
  auto scalar() const -> const ray_cast_result<default_index_t, Real> & {
    return std::get<ray_cast_result<default_index_t, Real>>(_value);
  }
  auto batch() -> ray_cast_primitive_batch_result<Real> & {
    return std::get<ray_cast_primitive_batch_result<Real>>(_value);
  }
  auto batch() const -> const ray_cast_primitive_batch_result<Real> & {
    return std::get<ray_cast_primitive_batch_result<Real>>(_value);
  }
};

/// @brief Scalar-or-batch result of casting runtime rays against a form.
template <typename Index, typename Real> class ray_cast_form_result {
  using scalar_type = ray_cast_result<Index, Real>;
  using batch_type = ray_cast_form_batch_result<Index, Real>;
  std::variant<scalar_type, batch_type> _value;

public:
  ray_cast_form_result(scalar_type value) : _value(std::move(value)) {}
  ray_cast_form_result(batch_type value) : _value(std::move(value)) {}

  auto is_scalar() const -> bool {
    return std::holds_alternative<scalar_type>(_value);
  }
  auto is_batch() const -> bool { return !is_scalar(); }
  auto scalar() -> scalar_type & { return std::get<scalar_type>(_value); }
  auto scalar() const -> const scalar_type & {
    return std::get<scalar_type>(_value);
  }
  auto batch() -> batch_type & { return std::get<batch_type>(_value); }
  auto batch() const -> const batch_type & {
    return std::get<batch_type>(_value);
  }
};

/// @brief Cast one ray or a ray batch against one primitive or a primitive
/// batch. Mixed primitive precision promotes to double. Two batches are paired
/// element-wise and must have equal lengths; otherwise the scalar side is
/// broadcast.
template <typename RayReal, typename TargetReal, std::size_t Dims>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const primitive<TargetReal, Dims> &target,
              const ray_cast_options<std::common_type_t<RayReal, TargetReal>>
                  &options = {})
    -> ray_cast_primitive_result<std::common_type_t<RayReal, TargetReal>>;

/// @brief Cast one Dims-dimensional ray or ray batch against a fixed- or
/// dynamic-connectivity mesh. Element IDs preserve the mesh index type.
template <typename RayReal, typename Index, typename FormReal, std::size_t Dims,
          std::size_t Ngon>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const mesh<Index, FormReal, Dims, Ngon> &form,
              const ray_cast_options<FormReal> &options = {})
    -> ray_cast_form_result<Index, FormReal>;

/// @brief Cast one Dims-dimensional ray or ray batch against a point cloud.
template <typename RayReal, typename FormReal, std::size_t Dims>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const point_cloud<FormReal, Dims> &form,
              const ray_cast_options<FormReal> &options = {})
    -> ray_cast_form_result<std::int32_t, FormReal>;

/// @brief Cast one Dims-dimensional ray or ray batch against an edge mesh.
/// Element IDs preserve the edge-mesh index type.
template <typename RayReal, typename Index, typename FormReal, std::size_t Dims>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const edge_mesh<Index, FormReal, Dims> &form,
              const ray_cast_options<FormReal> &options = {})
    -> ray_cast_form_result<Index, FormReal>;

#define TF_CPP_DECLARE_RAY_CAST_PRIMITIVE(RayReal, TargetReal, Dims)           \
  extern template auto ray_cast<RayReal, TargetReal, Dims>(                    \
      const primitive<RayReal, Dims> &, const primitive<TargetReal, Dims> &,   \
      const ray_cast_options<std::common_type_t<RayReal, TargetReal>> &)       \
      -> ray_cast_primitive_result<std::common_type_t<RayReal, TargetReal>>

TF_CPP_MATRIX_FOR_EACH_REAL_PAIR_DIMS(TF_CPP_DECLARE_RAY_CAST_PRIMITIVE)

#undef TF_CPP_DECLARE_RAY_CAST_PRIMITIVE

/// The ray's real is the query operand's own axis of the matrix, so it leads
/// the form's index and real exactly as the operands are written.
#define TF_CPP_DECLARE_RAY_CAST_MESH(RayReal, Index, FormReal, Dims, Ngon)     \
  extern template auto ray_cast<RayReal, Index, FormReal, Dims, Ngon>(         \
      const primitive<RayReal, Dims> &,                                        \
      const mesh<Index, FormReal, Dims, Ngon> &,                               \
      const ray_cast_options<FormReal> &)                                      \
      -> ray_cast_form_result<Index, FormReal>

#define TF_CPP_DECLARE_RAY_CAST_POINT_CLOUD(RayReal, FormReal, Dims)           \
  extern template auto ray_cast<RayReal, FormReal, Dims>(                      \
      const primitive<RayReal, Dims> &, const point_cloud<FormReal, Dims> &,   \
      const ray_cast_options<FormReal> &)                                      \
      -> ray_cast_form_result<std::int32_t, FormReal>

#define TF_CPP_DECLARE_RAY_CAST_EDGE_MESH(RayReal, Index, FormReal, Dims)      \
  extern template auto ray_cast<RayReal, Index, FormReal, Dims>(               \
      const primitive<RayReal, Dims> &,                                        \
      const edge_mesh<Index, FormReal, Dims> &,                                \
      const ray_cast_options<FormReal> &)                                      \
      -> ray_cast_form_result<Index, FormReal>

TF_CPP_MATRIX_FOR_EACH_REAL_INDEX_REAL_DIMS_NGON(TF_CPP_DECLARE_RAY_CAST_MESH)
TF_CPP_MATRIX_FOR_EACH_REAL_PAIR_DIMS(TF_CPP_DECLARE_RAY_CAST_POINT_CLOUD)
TF_CPP_MATRIX_FOR_EACH_REAL_INDEX_REAL_DIMS(TF_CPP_DECLARE_RAY_CAST_EDGE_MESH)

#undef TF_CPP_DECLARE_RAY_CAST_EDGE_MESH
#undef TF_CPP_DECLARE_RAY_CAST_POINT_CLOUD
#undef TF_CPP_DECLARE_RAY_CAST_MESH

} // namespace tf::cpp
