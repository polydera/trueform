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

#include "trueform/cpp/geometry/chamfer_error.hpp"
#include "trueform/cpp/geometry/chamfer_error_options.hpp"
#include "trueform/cpp/geometry/fit_icp.hpp"
#include "trueform/cpp/geometry/fit_knn.hpp"
#include "trueform/cpp/geometry/fit_obb.hpp"
#include "trueform/cpp/geometry/fit_rigid.hpp"
#include "trueform/cpp/geometry/symmetric_chamfer_error.hpp"

#include "trueform/core/policy/frame.hpp"
#include "trueform/core/policy/normals.hpp"
#include "trueform/cpp/core/to_matrix.hpp"
#include "trueform/geometry/chamfer_error.hpp"
#include "trueform/geometry/fit_icp_alignment.hpp"
#include "trueform/geometry/fit_knn_alignment.hpp"
#include "trueform/geometry/fit_obb_alignment.hpp"
#include "trueform/geometry/fit_rigid_alignment.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tf::cpp {
namespace registration_detail {

inline auto require_non_negative(int value, const char *name) -> void {
  if (value < 0)
    throw std::invalid_argument(std::string("registration: ") + name +
                                " must be non-negative");
}

template <typename Real>
auto require_finite(Real value, const char *name) -> void {
  if (!std::isfinite(value) || !std::isfinite(static_cast<float>(value)))
    throw std::invalid_argument(std::string("registration: ") + name +
                                " must be finite");
}

template <typename Real> auto require_outlier_proportion(Real value) -> void {
  require_finite(value, "outlier_proportion");
  if (value < Real{0} || value >= Real{1})
    throw std::invalid_argument(
        "registration: outlier_proportion must be in [0, 1)");
}

template <typename Real> auto require_ema_alpha(Real value) -> void {
  require_finite(value, "ema_alpha");
  if (value < Real{0} || value > Real{1})
    throw std::invalid_argument("registration: ema_alpha must be in [0, 1]");
}

/// The owner states its points' shape and drops normals a new point set does
/// not name, so an operand is refused for what it holds, not for how it holds
/// it: a cloud with no points has no alignment to fit.
template <typename Real, std::size_t Dims>
auto require_cloud(const point_cloud<Real, Dims> &value, const char *name)
    -> void {
  if (value.number_of_points() == 0)
    throw std::invalid_argument(std::string("registration: ") + name +
                                " point cloud must not be empty");
}

/// Normals weight a 3D fit; a 2D fit has no use for them, so an operand
/// carrying them is asking for an operation this entry does not have.
template <typename Real, std::size_t Dims>
auto require_fittable_cloud(const point_cloud<Real, Dims> &value,
                            const char *name) -> void {
  require_cloud(value, name);
  if (Dims == 2 && value.has_normals())
    throw std::invalid_argument(
        std::string("registration: ") + name +
        " normals are not supported for 2D registration");
}

/// points | tree | normals | frame — the order every operand in trueform is
/// tagged in, with the optionals this fit reads. The frame is always there, so
/// a cloud authored where it stands is tagged like one that was moved.
template <bool WithTree, bool WithNormals, typename Real, std::size_t Dims>
auto tagged_cloud(const point_cloud<Real, Dims> &value) {
  if constexpr (WithTree && WithNormals)
    return value.points() | tf::tag(value.tree()) |
           tf::tag_normals(value.normals()) | tf::tag(value.frame());
  else if constexpr (WithTree)
    return value.points() | tf::tag(value.tree()) | tf::tag(value.frame());
  else if constexpr (WithNormals)
    return value.points() | tf::tag_normals(value.normals()) |
           tf::tag(value.frame());
  else
    return value.points() | tf::tag(value.frame());
}

/// The target's normals select point-to-plane fitting and the source's then
/// weight it, so a source has nothing to weight until the target states one.
template <bool TargetNeedsTree, typename Real, std::size_t Dims, typename Fn>
auto with_optional_normals(const point_cloud<Real, Dims> &source,
                           const point_cloud<Real, Dims> &target, Fn &&fn)
    -> nd_array<Real> {
  if constexpr (Dims == 3) {
    if (target.has_normals()) {
      if (source.has_normals())
        return tf::cpp::to_matrix<Real>(
            fn(tagged_cloud<false, true>(source),
               tagged_cloud<TargetNeedsTree, true>(target)));
      return tf::cpp::to_matrix<Real>(
          fn(tagged_cloud<false, false>(source),
             tagged_cloud<TargetNeedsTree, true>(target)));
    }
  }
  return tf::cpp::to_matrix<Real>(
      fn(tagged_cloud<false, false>(source),
         tagged_cloud<TargetNeedsTree, false>(target)));
}

template <typename Real, std::size_t Dims>
auto directed_chamfer_error(const point_cloud<Real, Dims> &source,
                            const point_cloud<Real, Dims> &target,
                            Real outlier_proportion) -> Real {
  return tf::chamfer_error(tagged_cloud<false, false>(source),
                           tagged_cloud<true, false>(target),
                           static_cast<float>(outlier_proportion));
}

} // namespace registration_detail

template <typename Real, std::size_t Dims>
auto fit_icp(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_icp_options<Real> &options) -> nd_array<Real> {
  registration_detail::require_non_negative(options.max_iterations,
                                            "max_iterations");
  registration_detail::require_non_negative(options.n_samples, "n_samples");
  registration_detail::require_non_negative(options.k, "k");
  if (options.max_iterations > 0 && options.k == 0)
    throw std::invalid_argument(
        "registration: k must be positive when max_iterations is positive");
  registration_detail::require_finite(options.min_relative_improvement,
                                      "min_relative_improvement");
  registration_detail::require_ema_alpha(options.ema_alpha);
  registration_detail::require_finite(options.sigma, "sigma");
  registration_detail::require_outlier_proportion(options.outlier_proportion);
  registration_detail::require_fittable_cloud(source, "source");
  registration_detail::require_fittable_cloud(target, "target");

  const tf::icp_config config{
      static_cast<std::size_t>(options.max_iterations),
      static_cast<float>(options.min_relative_improvement),
      static_cast<float>(options.ema_alpha),
      static_cast<std::size_t>(options.n_samples),
      static_cast<std::size_t>(options.k),
      static_cast<float>(options.sigma),
      static_cast<float>(options.outlier_proportion),
  };
  return registration_detail::with_optional_normals<true>(
      source, target, [&](const auto &source_form, const auto &target_form) {
        return tf::fit_icp_alignment(source_form, target_form, config);
      });
}

template <typename Real, std::size_t Dims>
auto fit_rigid(const point_cloud<Real, Dims> &source,
               const point_cloud<Real, Dims> &target) -> nd_array<Real> {
  registration_detail::require_fittable_cloud(source, "source");
  registration_detail::require_fittable_cloud(target, "target");
  if (source.number_of_points() != target.number_of_points())
    throw std::invalid_argument(
        "registration: rigid alignment point counts must match");
  return registration_detail::with_optional_normals<false>(
      source, target, [](const auto &source_form, const auto &target_form) {
        return tf::fit_rigid_alignment(source_form, target_form);
      });
}

template <typename Real, std::size_t Dims>
auto fit_knn(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_knn_options<Real> &options) -> nd_array<Real> {
  if (options.k <= 0)
    throw std::invalid_argument("registration: k must be positive");
  registration_detail::require_finite(options.sigma, "sigma");
  registration_detail::require_outlier_proportion(options.outlier_proportion);
  registration_detail::require_fittable_cloud(source, "source");
  registration_detail::require_fittable_cloud(target, "target");

  const tf::knn_alignment_config config{
      static_cast<std::size_t>(options.k),
      static_cast<float>(options.sigma),
      static_cast<float>(options.outlier_proportion),
  };
  return registration_detail::with_optional_normals<true>(
      source, target, [&](const auto &source_form, const auto &target_form) {
        return tf::fit_knn_alignment(source_form, target_form, config);
      });
}

template <typename Real, std::size_t Dims>
auto fit_obb(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_obb_options &options) -> nd_array<Real> {
  registration_detail::require_non_negative(options.sample_size, "sample_size");
  registration_detail::require_cloud(source, "source");
  registration_detail::require_cloud(target, "target");

  return tf::cpp::to_matrix<Real>(tf::fit_obb_alignment(
      registration_detail::tagged_cloud<false, false>(source),
      registration_detail::tagged_cloud<true, false>(target),
      static_cast<std::size_t>(options.sample_size)));
}

template <typename Real, std::size_t Dims>
auto chamfer_error(const point_cloud<Real, Dims> &source,
                   const point_cloud<Real, Dims> &target,
                   const chamfer_error_options<Real> &options) -> Real {
  registration_detail::require_outlier_proportion(options.outlier_proportion);
  registration_detail::require_cloud(source, "source");
  registration_detail::require_cloud(target, "target");
  return registration_detail::directed_chamfer_error(
      source, target, options.outlier_proportion);
}

template <typename Real, std::size_t Dims>
auto symmetric_chamfer_error(const point_cloud<Real, Dims> &source,
                             const point_cloud<Real, Dims> &target,
                             const chamfer_error_options<Real> &options)
    -> Real {
  registration_detail::require_outlier_proportion(options.outlier_proportion);
  registration_detail::require_cloud(source, "source");
  registration_detail::require_cloud(target, "target");
  const auto source_to_target = registration_detail::directed_chamfer_error(
      source, target, options.outlier_proportion);
  const auto target_to_source = registration_detail::directed_chamfer_error(
      target, source, options.outlier_proportion);
  return (source_to_target + target_to_source) / Real{2};
}

} // namespace tf::cpp
