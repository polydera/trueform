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

#include "trueform/cpp/remesh/decimated.hpp"
#include "trueform/cpp/remesh/isotropic_remeshed.hpp"
#include "trueform/cpp/remesh/simplified.hpp"

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/angle.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/remesh/decimated.hpp"
#include "trueform/remesh/isotropic_remeshed.hpp"
#include "trueform/remesh/simplified.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

inline auto empty_regions() -> nd_array<std::int32_t> {
  tf::buffer<std::int32_t> buffer;
  buffer.allocate(0);
  return nd_array<std::int32_t>::from_buffer(std::move(buffer), {0});
}

inline auto require_regions(const nd_array<std::int32_t> &regions,
                            std::size_t number_of_faces, const char *operation)
    -> void {
  if (!regions.is_valid())
    return;
  if (regions.ndim() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": regions must be one-dimensional");
  if (regions.empty())
    return;
  if (regions.length() != number_of_faces)
    throw std::invalid_argument(std::string(operation) +
                                ": regions must have one label per face");
}

template <typename Real>
auto require_collapse_config(const tf::collapse_guard_config<Real> &config,
                             const char *operation) -> void {
  if (!std::isfinite(config.min_quality) || config.min_quality > Real{1})
    throw std::invalid_argument(std::string(operation) +
                                ": min quality must be finite and at most 1");
  if (!std::isfinite(config.feature_angle.value))
    throw std::invalid_argument(std::string(operation) +
                                ": feature angle must be finite");
  if (!std::isfinite(config.feature_weight) || config.feature_weight < Real{0})
    throw std::invalid_argument(
        std::string(operation) +
        ": feature weight must be finite and nonnegative");
  if (!std::isfinite(config.stabilizer) || config.stabilizer < 0)
    throw std::invalid_argument(std::string(operation) +
                                ": stabilizer must be finite and nonnegative");
}

template <typename Real>
auto require_decimate_config(Real target_proportion,
                             const tf::decimate_config<Real> &config) -> void {
  if (!std::isfinite(target_proportion) || target_proportion < Real{0} ||
      target_proportion > Real{1})
    throw std::invalid_argument(
        "decimated: target proportion must be finite and in [0, 1]");
  require_collapse_config(config, "decimated");
}

template <typename Real>
auto require_isotropic_config(const tf::isotropic_remesh_config<Real> &config)
    -> void {
  if (!std::isfinite(config.target_length) || config.target_length <= Real{0})
    throw std::invalid_argument(
        "isotropic_remeshed: target length must be finite and positive");
  if (config.iterations < 0 || config.relaxation_iters < 0)
    throw std::invalid_argument(
        "isotropic_remeshed: iteration counts must be nonnegative");
  if (!std::isfinite(config.lambda) || config.lambda <= Real{0} ||
      config.lambda > Real{1})
    throw std::invalid_argument(
        "isotropic_remeshed: lambda must be finite and in (0, 1]");
  require_collapse_config(config, "isotropic_remeshed");
}

template <typename Real>
auto require_simplify_config(const tf::simplify_config<Real> &config) -> void {
  if (!std::isfinite(config.error_rel) || config.error_rel < Real{0})
    throw std::invalid_argument(
        "simplified: relative error must be finite and nonnegative");
  if (config.iterations < 0 || config.optimize_iterations < 0 ||
      config.relaxation_iters < 0)
    throw std::invalid_argument(
        "simplified: iteration counts must be nonnegative");
  if (!std::isfinite(config.lambda) || config.lambda <= Real{0} ||
      config.lambda > Real{1})
    throw std::invalid_argument(
        "simplified: lambda must be finite and in (0, 1]");
  require_collapse_config(config, "simplified");
}

template <typename Index, typename Real, typename Form>
auto empty_remesh_result(const Form &form) -> remesh_result<Index, Real> {
  tf::polygons_buffer<Index, Real, 3, 3> polygons;
  polygons.faces_buffer().allocate(0);
  polygons.points_buffer().allocate(form.points().size());
  tf::parallel_copy(form.points(), polygons.points());
  return {std::move(polygons), empty_regions()};
}

template <typename Index, typename Real, typename Labels>
auto make_regions_result(tf::polygons_buffer<Index, Real, 3, 3> &&polygons,
                         Labels &&labels) -> remesh_result<Index, Real> {
  const auto count = static_cast<int>(labels.size());
  return {std::move(polygons), nd_array<std::int32_t>::from_buffer(
                                   std::forward<Labels>(labels), {count})};
}

template <typename Index, typename Real, typename Form>
auto run_decimated(const Form &form, Real target_proportion,
                   const tf::decimate_config<Real> &config,
                   const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  if (!form.size())
    return empty_remesh_result<Index, Real>(form);
  if (regions.is_valid() && !regions.empty()) {
    auto [polygons, half_edges, labels] =
        tf::decimated(form, target_proportion, config,
                      tf::preserve_regions(regions.make_range()));
    return make_regions_result<Index, Real>(std::move(polygons),
                                            std::move(labels));
  }
  auto [polygons, half_edges] = tf::decimated(form, target_proportion, config);
  return {std::move(polygons), empty_regions()};
}

template <typename Index, typename Real, typename Form>
auto run_isotropic_remeshed(const Form &form,
                            const tf::isotropic_remesh_config<Real> &config,
                            const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  if (!form.size())
    return empty_remesh_result<Index, Real>(form);
  if (regions.is_valid() && !regions.empty()) {
    auto [polygons, half_edges, labels] = tf::isotropic_remeshed(
        form, config, tf::preserve_regions(regions.make_range()));
    return make_regions_result<Index, Real>(std::move(polygons),
                                            std::move(labels));
  }
  auto [polygons, half_edges] = tf::isotropic_remeshed(form, config);
  return {std::move(polygons), empty_regions()};
}

template <typename Index, typename Real, typename Form>
auto run_simplified(const Form &form, const tf::simplify_config<Real> &config,
                    const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  if (!form.size())
    return empty_remesh_result<Index, Real>(form);
  if (regions.is_valid() && !regions.empty()) {
    auto [polygons, half_edges, labels] = tf::simplified(
        form, config, tf::preserve_regions(regions.make_range()));
    return make_regions_result<Index, Real>(std::move(polygons),
                                            std::move(labels));
  }
  auto [polygons, half_edges] = tf::simplified(form, config);
  return {std::move(polygons), empty_regions()};
}

/// A remesh walks and rewires the topology, so it takes the mesh with the
/// structures that walk: the membership and the half edges, both the cache's.
template <typename Index, typename Real>
auto remesh_form(const mesh<Index, Real, 3> &value) {
  return value.polygons() | tf::tag(value.face_membership()) |
         tf::tag(value.half_edges()) | tf::tag(value.frame());
}

} // namespace detail

template <typename Index, typename Real>
auto decimated(const mesh<Index, Real, 3> &value, Real target_proportion,
               tf::decimate_config<Real> config,
               const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  detail::require_decimate_config(target_proportion, config);
  value.require_indices();
  detail::require_regions(regions, value.number_of_faces(), "decimated");
  return detail::run_decimated<Index, Real>(detail::remesh_form(value),
                                            target_proportion, config, regions);
}

template <typename Index, typename Real>
auto isotropic_remeshed(const mesh<Index, Real, 3> &value,
                        tf::isotropic_remesh_config<Real> config,
                        const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  detail::require_isotropic_config(config);
  value.require_indices();
  detail::require_regions(regions, value.number_of_faces(),
                          "isotropic_remeshed");
  return detail::run_isotropic_remeshed<Index, Real>(detail::remesh_form(value),
                                                     config, regions);
}

template <typename Index, typename Real>
auto simplified(const mesh<Index, Real, 3> &value,
                tf::simplify_config<Real> config,
                const nd_array<std::int32_t> &regions)
    -> remesh_result<Index, Real> {
  detail::require_simplify_config(config);
  value.require_indices();
  detail::require_regions(regions, value.number_of_faces(), "simplified");
  return detail::run_simplified<Index, Real>(detail::remesh_form(value), config,
                                             regions);
}

} // namespace tf::cpp
