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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/curves_buffer.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../intersect/intersect_config.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/embedded_intersection_curves.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto embedded_intersection_curves(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using InputReal =
            tf::coordinate_type<std::decay_t<decltype(p0)>,
                                std::decay_t<decltype(p1)>>;
        using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
        using PipelineReal =
            std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
        using RealOut =
            std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                               InputReal, OutputCoordinateType>;
        auto [ibp, ig, fc, cg] = cut::dispatch::build_exact_pipeline<
            Index, PipelineReal, ResolvedInt>(p0, p1, config);
        auto [res, fl] =
            tf::cut::embedded_intersection_curves<Index, OutputCoordinateType>(
                p0, ig, fc, ibp.converter(), Index(0));
        if constexpr (!std::is_integral_v<InputReal> &&
                      std::is_integral_v<RealOut>) {
          auto conv = ibp.converter();
          return std::make_tuple(std::move(res), std::move(fl),
                                 std::move(conv));
        } else {
          return std::make_tuple(std::move(res), std::move(fl));
        }
      });
}

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1,
                                  tf::return_curves_t) {
  return embedded_intersection_curves<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_config{}, tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1,
                                  tf::intersect_config config,
                                  tf::return_curves_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using InputReal =
            tf::coordinate_type<std::decay_t<decltype(p0)>,
                                std::decay_t<decltype(p1)>>;
        using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
        using PipelineReal =
            std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
        using RealOut =
            std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                               InputReal, OutputCoordinateType>;
        auto [ibp, ig, fc, cg] = cut::dispatch::build_exact_pipeline<
            Index, PipelineReal, ResolvedInt>(p0, p1, config);
        auto [res, fl] =
            tf::cut::embedded_intersection_curves<Index, OutputCoordinateType>(
                p0, ig, fc, ibp.converter(), Index(0));

        auto paths =
            tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
        auto &conv = ibp.converter();
        auto ipts = ig.points();
        tf::curves_buffer<Index, RealOut, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        if constexpr (std::is_integral_v<RealOut>) {
          tf::parallel_copy(tf::make_points(ipts), cb.points());
        } else {
          tf::parallel_copy(
              tf::make_points(tf::make_mapped_range(
                  ipts,
                  [&conv](const auto &pt) { return conv.deconvert(pt); })),
              cb.points());
        }
        if constexpr (!std::is_integral_v<InputReal> &&
                      std::is_integral_v<RealOut>) {
          auto conv_copy = ibp.converter();
          return std::make_tuple(std::move(res), std::move(fl), std::move(cb),
                                 std::move(conv_copy));
        } else {
          return std::make_tuple(std::move(res), std::move(fl), std::move(cb));
        }
      });
}

} // namespace tf
