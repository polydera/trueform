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
#include "../intersect/exact/make_kernel.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/intersect_config.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../reindex/return_index_map.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/make_mesh_arrangement_index_map.hpp"
#include "./construct/make_mesh_arrangements.hpp"
#include "./cut_graph.hpp"
#include "./dispatch/arrangement.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./face_cuts.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace cut {

/// Shared worker for the two-mesh arrangement entry points. `Curves` /
/// `IndexMap` are each `tf::none_t` (skip) or their request tag; the returned
/// tuple grows accordingly. Body lives once; public overloads dispatch here.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P0, typename P1>
auto mesh_arrangements_pair_worker(const P0 &p0, const P1 &p1,
                                   tf::intersect_config config) {
  using Index =
      std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                         typename std::decay_t<decltype(p1)>::index_type>;
  using InputReal = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                        std::decay_t<decltype(p1)>>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;

  auto [ibp, ig, fc, cg] =
      cut::dispatch::build_exact_pipeline<Index, PipelineReal, ResolvedInt>(
          p0, p1, config);
  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<OutputCoordinateType>(ig, fc, p0, p1,
                                                            ibp.converter());

  if constexpr (want_imap) {
    auto imap = tf::cut::make_mesh_arrangement_index_map(
        std::move(map_data), std::move(tag_labels), std::move(face_labels),
        static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap), ibp.converter());
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
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
              ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
          cb.points());
    }
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb),
                             ibp.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), ibp.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels));
  }
}

/// Shared worker for the N-mesh arrangement entry point. `Curves`/`IndexMap`
/// default to `tf::none_t` (plain), so existing 2-arg-template callers keep
/// working. Folds the curve build (SoS or cut-graph) and the index map into
/// one body.
template <typename Int, typename OutputCoordinateType,
          typename Curves = tf::none_t, typename IndexMap = tf::none_t,
          typename FormsRange>
auto mesh_arrangements_n(const FormsRange &forms, tf::intersect_config config) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;

  tf::intersections_between_polygons<Index, PipelineReal, ResolvedInt> ibp;
  ibp.build(forms, config);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<ResolvedInt, 3> {
    return conv.convert(
        tf::transformed(forms[tag].points()[id], tf::frame_of(forms[tag])));
  };

  tf::intersection_graph<Index, ResolvedInt> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, config.mode,
           tf::exact::make_kernel(conv, config.tolerance));

  tf::face_cuts<Index, ResolvedInt> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<RealOut>(ig, fc, forms, conv);

  if constexpr (want_imap) {
    auto imap = make_mesh_arrangement_index_map(
        std::move(map_data), std::move(tag_labels), std::move(face_labels),
        static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap), ibp.converter());
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
    auto paths = [&]() {
      if (config.mode & tf::intersect_mode::sos) {
        auto edge_pairs = tf::make_mapped_range(
            ig.edge_groups(), [](const auto &group) -> std::array<Index, 2> {
              return {group[0].point_0, group[0].point_1};
            });
        return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
      } else {
        tf::cut_graph<Index> cg;
        cg.build(fc, static_cast<Index>(ig.points().size()));
        return tf::connect_edges_to_paths(
            tf::make_edges(cg.intersection_edges()));
      }
    }();
    tf::curves_buffer<Index, RealOut, 3> cb;
    cb.paths_buffer() = std::move(paths);
    auto ipts = ig.points();
    cb.points_buffer().allocate(ipts.size());
    if constexpr (std::is_integral_v<RealOut>) {
      tf::parallel_copy(tf::make_points(ipts), cb.points());
    } else {
      tf::parallel_copy(
          tf::make_points(tf::make_mapped_range(
              ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
          cb.points());
    }
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb),
                             ibp.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), ibp.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels));
  }
}

} // namespace cut

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::intersect_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        return cut::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                                  tf::none_t, tf::none_t>(
            p0, p1, config);
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::intersect_config config, tf::return_curves_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        return cut::mesh_arrangements_pair_worker<
            Int, OutputCoordinateType, tf::return_curves_t, tf::none_t>(p0, p1,
                                                                        config);
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curves (default intersect mode).
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_config{}, tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes, returning a @ref tf::mesh_arrangement_index_map
///        relating the output back to the input meshes.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::intersect_config config,
                            tf::return_index_map_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        return cut::mesh_arrangements_pair_worker<
            Int, OutputCoordinateType, tf::none_t, tf::return_index_map_t>(
            p0, p1, config);
      });
}

/// @ingroup cut_boolean
/// @brief Two-mesh index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_index_map_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_config{}, tf::return_index_map);
}

/// @ingroup cut_boolean
/// @brief Build a single merged mesh from N intersected meshes.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(
    const Range &_forms,
    tf::intersect_config config = {tf::intersect_mode::primitives |
                                   tf::intersect_mode::resolve_crossing_contours}) {
  return cut::dispatch::arrangement(_forms, [config](const auto &forms) {
    return cut::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                    tf::none_t>(forms, config);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with intersection curves.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &_forms, tf::intersect_config config,
                            tf::return_curves_t) {
  return cut::dispatch::arrangement(_forms, [config](const auto &forms) {
    return cut::mesh_arrangements_n<Int, OutputCoordinateType,
                                    tf::return_curves_t, tf::none_t>(forms,
                                                                     config);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with curves (default intersect mode).
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      forms,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours},
      tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh from N meshes, returning a
///        @ref tf::mesh_arrangement_index_map relating output back to input.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &_forms, tf::intersect_config config,
                            tf::return_index_map_t) {
  return cut::dispatch::arrangement(_forms, [config](const auto &forms) {
    return cut::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                    tf::return_index_map_t>(forms, config);
  });
}

/// @ingroup cut_boolean
/// @brief N-mesh index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_index_map_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      forms,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours},
      tf::return_index_map);
}

} // namespace tf
