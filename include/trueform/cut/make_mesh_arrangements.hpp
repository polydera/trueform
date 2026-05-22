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
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/make_mesh_arrangements.hpp"
#include "./cut_graph.hpp"
#include "./dispatch/arrangement.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./face_cuts.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_mode mode = tf::intersect_mode::primitives) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [mode](const auto &p0, const auto &p1) {
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
            Index, PipelineReal, ResolvedInt>(p0, p1, mode);
        auto [mesh, tag_labels, face_labels, map_data] =
            tf::cut::make_mesh_arrangements<OutputCoordinateType>(
                ig, fc, p0, p1, ibp.converter());
        if constexpr (!std::is_integral_v<InputReal> &&
                      std::is_integral_v<RealOut>) {
          auto conv = ibp.converter();
          return std::make_tuple(std::move(mesh), std::move(tag_labels),
                                 std::move(face_labels), std::move(conv));
        } else {
          return std::make_tuple(std::move(mesh), std::move(tag_labels),
                                 std::move(face_labels));
        }
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curve output.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::intersect_mode mode, tf::return_curves_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [mode](const auto &p0, const auto &p1) {
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
            Index, PipelineReal, ResolvedInt>(p0, p1, mode);
        auto [mesh, tag_labels, face_labels, map_data] =
            tf::cut::make_mesh_arrangements<OutputCoordinateType>(
                ig, fc, p0, p1, ibp.converter());

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
          return std::make_tuple(std::move(mesh), std::move(tag_labels),
                                 std::move(face_labels), std::move(cb),
                                 std::move(conv_copy));
        } else {
          return std::make_tuple(std::move(mesh), std::move(tag_labels),
                                 std::move(face_labels), std::move(cb));
        }
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curves (default intersect mode).
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_mode::primitives,
      tf::return_curves);
}

namespace cut {

template <typename Int, typename OutputCoordinateType, typename FormsRange>
auto mesh_arrangements_n(const FormsRange &forms, tf::intersect_mode mode) {
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

  // Internal converter is parameterised on PipelineReal: double for
  // floating-point input (keeps deconverted intermediates at full
  // precision), or the integral input type itself (yields an identity
  // converter, no rescale). Output points buffer is allocated at RealOut.
  tf::intersections_between_polygons<Index, PipelineReal, ResolvedInt> ibp;
  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<ResolvedInt, 3> {
    return conv.convert(
        tf::transformed(forms[tag].points()[id], tf::frame_of(forms[tag])));
  };

  tf::intersection_graph<Index, ResolvedInt> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, mode);

  tf::face_cuts<Index, ResolvedInt> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<RealOut>(ig, fc, forms, conv);

  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    auto conv_copy = ibp.converter();
    return std::make_tuple(std::move(mesh), std::move(tag_labels),
                           std::move(face_labels), std::move(conv_copy));
  } else {
    return std::make_tuple(std::move(mesh), std::move(tag_labels),
                           std::move(face_labels));
  }
}

template <typename Int, typename OutputCoordinateType, typename FormsRange>
auto mesh_arrangements_n_curves(const FormsRange &forms,
                                tf::intersect_mode mode) {
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

  tf::intersections_between_polygons<Index, PipelineReal, ResolvedInt> ibp;
  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<ResolvedInt, 3> {
    return conv.convert(
        tf::transformed(forms[tag].points()[id], tf::frame_of(forms[tag])));
  };

  tf::intersection_graph<Index, ResolvedInt> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, mode);

  tf::face_cuts<Index, ResolvedInt> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<RealOut>(ig, fc, forms, conv);

  auto paths = [&]() {
    if (mode & tf::intersect_mode::sos) {
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

  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    auto conv_copy = ibp.converter();
    return std::make_tuple(std::move(mesh), std::move(tag_labels),
                           std::move(face_labels), std::move(cb),
                           std::move(conv_copy));
  } else {
    return std::make_tuple(std::move(mesh), std::move(tag_labels),
                           std::move(face_labels), std::move(cb));
  }
}

} // namespace cut

/// @ingroup cut_boolean
/// @brief Build a single merged mesh from N intersected meshes.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(
    const Range &_forms,
    tf::intersect_mode mode = tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours) {
  return cut::dispatch::arrangement(_forms, [mode](const auto &forms) {
    return cut::mesh_arrangements_n<Int, OutputCoordinateType>(forms, mode);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with intersection curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &_forms, tf::intersect_mode mode,
                            tf::return_curves_t) {
  return cut::dispatch::arrangement(_forms, [mode](const auto &forms) {
    return cut::mesh_arrangements_n_curves<Int, OutputCoordinateType>(forms,
                                                                       mode);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with curves (default intersect mode).
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      forms,
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours,
      tf::return_curves);
}

} // namespace tf
