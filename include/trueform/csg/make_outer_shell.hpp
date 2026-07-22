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
#include "../clean/polygons.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../cut/make_polygon_arrangements.hpp"
#include "../topology/domain_config.hpp"
#include "../topology/make_domain_labels.hpp"
#include "./csg_graph.hpp"
#include "./graph/compute_chosen_sides.hpp"
#include "./graph/make_csg_mesh.hpp"
#include "./graph/structural_membership.hpp"

namespace tf {

/// @ingroup csg
/// @brief Outer shell of a csg graph: the boundary of the union of
/// everything its forms enclose.
///
/// Structural read — no winding bits, no boolean expression. The
/// unbounded universe is the most-negative-volume domain (the seeder's
/// oracle, stored on the graph); the shell is the boundary between the
/// universe and everything else. This is the immersion-safe read: a
/// self-intersecting form's double-covered pockets stay enclosed, where
/// a winding-parity bit would drop them. The natural extraction for a
/// one-form graph (the self arrangement), valid for any N.
///
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, which resolves to the graph's input type.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_outer_shell(const tf::csg_graph<Forms, Structs, Int> &graph) {
  using InputReal =
      typename tf::csg_graph<Forms, Structs, Int>::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;

  auto membership = tf::csg::graph::compute_structural_membership(
      graph.domain_volumes(), graph.domain_nesting_merges());

  auto chosen =
      tf::csg::graph::compute_chosen_sides(graph.descriptor(), membership);
  return tf::csg::graph::make_csg_mesh<RealOut>(
      graph.labels(), graph.triangulations(), graph.created_points(),
      graph.forms(), chosen, graph.converter());
}

/// @ingroup csg
/// @brief Repair a mesh to its outer shell: the boundary of the union of
/// everything it encloses.
///
/// Splits the mesh at its self-intersection curves, labels the volumetric
/// domains, and keeps only the faces bounding the unbounded outside,
/// oriented outward. Internal structure — overlap membranes between
/// interpenetrating parts, faces buried inside the solid, enclosed
/// cavities — is removed. The result is free of self-intersections and
/// suitable as a boolean or csg-graph operand.
///
/// @tparam Int Exact integer type for the arrangement (deduced by default).
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param config The intersection mode flags for the self-arrangement.
/// @param domain_flags Flags forwarded to @ref tf::make_domain_labels.
///        `ignore_open_fragments` masks open fragments out of the region
///        formation; `exclude_outer_shell` is stripped (the extraction
///        needs the outer label).
/// @return A @ref tf::polygons_buffer bounding the enclosed union.
template <typename Int = tf::none_t, typename Policy>
auto make_outer_shell(const tf::polygons<Policy> &_polygons,
                      tf::intersect_config config = {tf::intersect_mode::primitives |
                                                     tf::intersect_mode::resolve_contours},
                      tf::domain_config domain_flags = tf::domain_config::none) {
  using Index = std::decay_t<decltype(_polygons.faces()[0][0])>;
  using RealT = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  constexpr auto N = tf::static_size_v<decltype(_polygons[0])>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, RealT>;

  auto [arranged, arrangement_labels, converter] =
      tf::make_polygon_arrangements<Int, ResolvedInt>(_polygons, config);
  (void)arrangement_labels;
  auto merged = tf::cleaned(arranged.polygons());
  auto polygons = merged.polygons();
  // The extraction keys on the outer label, so exclude_outer_shell must
  // not fold it onto the sentinel open fragments also map to.
  auto labeling_flags = static_cast<tf::domain_config>(
      static_cast<int>(domain_flags) &
      ~static_cast<int>(tf::domain_config::exclude_outer_shell));
  auto dl = tf::make_domain_labels<Int>(polygons, labeling_flags);

  // Emissions as in split_into_domains (2 * face + side), keeping only
  // faces with the outer shell on exactly one side. Orientation targets
  // outward-of-solid, the opposite of split_into_domains'
  // outward-of-domain rule: shell on side 0 (stored normal points into
  // the outside) keeps the winding, shell on side 1 reverses it. Faces
  // with the shell on both sides are dangling membranes — dropped.
  const auto &flat = dl.labels.data_buffer();
  auto shell = dl.outer_shell_label;
  tf::buffer<Index> kept;
  tf::generic_generate(
      tf::enumerate(polygons.faces()), kept,
      [&](const auto &pair, auto &out) {
        const auto &[f, face] = pair;
        bool s0 = flat[2 * f] == shell;
        bool s1 = flat[2 * f + 1] == shell;
        if (s0 != s1)
          out.push_back(static_cast<Index>(2 * f + (s1 ? 1 : 0)));
      });

  tf::polygons_buffer<Index, RealT, Dims, N> out;
  tf::buffer<Index> point_map;
  point_map.allocate(polygons.points().size());
  tf::parallel_fill(point_map, Index(-1));
  tf::buffer<Index> pt_ids;
  Index current = 0;
  for (auto elem_id : kept) {
    auto face = polygons.faces()[elem_id / 2];
    for (auto v_id : face) {
      auto &v = point_map[v_id];
      if (v == Index(-1)) {
        pt_ids.push_back(v_id);
        v = current++;
      }
    }
  }

  auto n_faces_out = static_cast<Index>(kept.size());
  if constexpr (N == tf::dynamic_size) {
    auto &offs = out.faces_buffer().offsets_buffer();
    offs.allocate(n_faces_out + 1);
    offs[0] = 0;
    std::size_t total = 0;
    for (auto [i, elem_id] : tf::enumerate(kept)) {
      total += polygons.faces()[elem_id / 2].size();
      offs[i + 1] = static_cast<Index>(total);
    }
    out.faces_buffer().data_buffer().allocate(total);
  } else {
    out.faces_buffer().data_buffer().allocate(std::size_t(N) * n_faces_out);
  }

  auto &data = out.faces_buffer().data_buffer();
  if constexpr (N == tf::dynamic_size) {
    auto &offs = out.faces_buffer().offsets_buffer();
    tf::parallel_for_each(tf::enumerate(kept), [&](auto pair) {
      auto &&[i, elem_id] = pair;
      bool reverse = (elem_id & Index(1)) == Index(1);
      auto face = polygons.faces()[elem_id / 2];
      auto n_v = face.size();
      Index write_at = offs[i];
      for (decltype(n_v) k = 0; k < n_v; ++k) {
        auto src_idx = reverse ? (n_v - 1 - k) : k;
        data[write_at + k] = point_map[face[src_idx]];
      }
    });
  } else {
    const Index stride = Index(N);
    tf::parallel_for_each(tf::enumerate(kept), [&](auto pair) {
      auto &&[i, elem_id] = pair;
      bool reverse = (elem_id & Index(1)) == Index(1);
      auto face = polygons.faces()[elem_id / 2];
      auto n_v = face.size();
      Index write_at = stride * Index(i);
      for (decltype(n_v) k = 0; k < n_v; ++k) {
        auto src_idx = reverse ? (n_v - 1 - k) : k;
        data[write_at + k] = point_map[face[src_idx]];
      }
    });
  }

  out.points_buffer().allocate(pt_ids.size());
  using out_point_t = tf::point<RealT, Dims>;
  using int_point_t = tf::point<ResolvedInt, Dims>;
  tf::parallel_copy(
      tf::make_points(tf::make_mapped_range(
          tf::make_indirect_range(pt_ids, polygons.points()),
          [&conv = converter](const auto &p) -> out_point_t {
            auto pt = conv.deconvert(int_point_t{p[0], p[1], p[2]});
            return {RealT(pt[0]), RealT(pt[1]), RealT(pt[2])};
          })),
      out.points_buffer());
  return out;
}

} // namespace tf
