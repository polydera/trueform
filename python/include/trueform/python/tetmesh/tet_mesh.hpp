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

#include "../arrangement/arrangement_builders.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/buffer.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/return_report.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>
#include <trueform/tetmesh/tet_mesh_config.hpp>
#include <trueform/tetmesh/tet_mesh_index_map.hpp>
#include <trueform/tetmesh/tet_mesh_report.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>
#include <utility>

namespace tf::py {

/// A tet mesh report as a dict of its fields, the census nested as a dict
/// and its face routes nested in that.
inline auto make_tet_mesh_report_dict(const tf::tet_mesh_report &report)
    -> nanobind::dict {
  nanobind::dict faces;
  const auto &f = report.census.faces;
  faces["covered"] = f.covered;
  faces["passes"] = f.passes;
  faces["flipped"] = f.flipped;
  faces["points"] = f.points;
  faces["unplaced"] = f.unplaced;
  faces["on_piece"] = f.on_piece;
  faces["refused"] = f.refused;
  faces["open_rim"] = f.open_rim;
  faces["crossed"] = f.crossed;
  faces["off_lattice"] = f.off_lattice;
  faces["budget"] = f.budget;
  faces["pierced"] = f.pierced;
  faces["standing"] = f.standing;
  nanobind::dict census;
  const auto &c = report.census;
  census["segment_rounds"] = c.segment_rounds;
  census["shell_splits"] = c.shell_splits;
  census["segments_refused"] = c.segments_refused;
  census["unrestored_facets"] = c.unrestored_facets;
  census["faces"] = faces;
  census["flip_recovered"] = c.flip_recovered;
  census["search_recovered"] = c.search_recovered;
  census["split_rows"] = c.split_rows;
  census["hull_barred"] = c.hull_barred;
  census["restored_before_faces"] = c.restored_before_faces;
  census["retired_crossing"] = c.retired_crossing;
  census["retired_neighbour"] = c.retired_neighbour;
  census["revived_segments"] = c.revived_segments;
  census["collapsed"] = c.collapsed;
  census["collapse_refused"] = c.collapse_refused;
  census["stalled_drains"] = c.stalled_drains;
  census["budget_drains"] = c.budget_drains;
  nanobind::dict out;
  out["status"] = int(report.status);
  out["refused_segments"] = report.refused_segments;
  out["refused_depth"] = report.refused_depth;
  out["refused_budget"] = report.refused_budget;
  out["refused_faces"] = report.refused_faces;
  out["rung"] = report.rung;
  out["drains"] = report.drains;
  out["pairs_named"] = report.pairs_named;
  out["pairs_cut"] = report.pairs_cut;
  out["triangles_cut"] = report.triangles_cut;
  out["dropped_degenerate"] = report.dropped_degenerate;
  out["stitched_triangles"] = report.stitched_triangles;
  out["failed_triangles"] = report.failed_triangles;
  out["pairs_unresolved"] = report.pairs_unresolved;
  out["input_open_edges"] = report.input_open_edges;
  out["recovered_open_edges"] = report.recovered_open_edges;
  out["storage_bits"] = report.storage_bits;
  out["input_bits"] = report.input_bits;
  out["frame_bits"] = report.frame_bits;
  out["deep_sites"] = report.deep_sites;
  out["deepest"] = report.deepest;
  out["corner_bits_needed"] = report.corner_bits_needed;
  out["min_angle"] = report.min_angle;
  out["flat"] = report.flat;
  out["inverted"] = report.inverted;
  out["negative_cells"] = report.negative_cells;
  out["census"] = census;
  return out;
}

/// A tet mesh index map's lanes: the facet triangles as (offsets, (N, 3)
/// rows) and the face facets as (offsets, data).
template <typename Index>
auto make_tet_mesh_index_map_tuple(tf::tet_mesh_index_map<Index> &&map) {
  const auto &rows = map.facet_triangles.data_buffer();
  tf::buffer<Index> flat;
  flat.allocate(rows.size() * 3);
  for (std::size_t k = 0; k < rows.size(); ++k)
    for (std::size_t c = 0; c < 3; ++c)
      flat[k * 3 + c] = rows[k][c];
  const std::size_t n_rows = rows.size();
  auto face_facets = make_numpy_array(std::move(map.face_facets));
  return nanobind::make_tuple(
      make_numpy_array(std::move(map.facet_triangles.offsets_buffer())),
      make_numpy_array(std::move(flat), {n_rows, 3}), face_facets.first,
      face_facets.second);
}

/// tf::make_tet_mesh over a triangle mesh: its crossings split, or without `split_constraints`
/// taken as given, its planes pooled under `tolerance`. The form carries the
/// mesh's face membership, manifold edge link and transformation, built and
/// cached where missing; the entrance reads no tree, so none is built.
/// Returns (tetrahedra, points, status) in double, the report in the
/// status's place when `return_report` asks for it, then the index map's
/// lanes when `return_index_map` asks for them.
template <typename Index, typename RealT>
auto tet_mesh_impl(mesh_wrapper<Index, RealT, 3, 3> &mesh, bool split_constraints,
                   double tolerance,
                   std::optional<std::size_t> site_budget_ratio,
                   bool return_report, bool return_index_map)
    -> nanobind::tuple {
  tf::tet_mesh_config config{split_constraints, tolerance};
  if (site_budget_ratio)
    config.site_budget_ratio = *site_budget_ratio;
  auto membership = mesh.compute_face_membership_if_missing();
  auto link = mesh.compute_manifold_edge_link_if_missing(membership);
  if (membership)
    mesh.commit_face_membership(std::move(*membership));
  if (link)
    mesh.commit_manifold_edge_link(std::move(*link));
  const auto form = mesh.make_primitive_range() |
                    tf::tag(mesh.face_membership()) |
                    tf::tag(mesh.manifold_edge_link()) |
                    tf::tag(form_transformation(mesh));
  const auto lanes = [](auto &tets, nanobind::object answer) {
    return nanobind::make_tuple(
        make_numpy_array(std::move(tets.tetras_buffer())),
        make_numpy_array(std::move(tets.points_buffer())), answer);
  };
  const auto with_map = [](nanobind::tuple head, auto &map) {
    return nanobind::make_tuple(head[0], head[1], head[2],
                                make_tet_mesh_index_map_tuple(std::move(map)));
  };
  if (return_report) {
    if (return_index_map) {
      auto [tets, report, map] = tf::make_tet_mesh(
          form, config, tf::return_report, tf::return_index_map);
      return with_map(lanes(tets, make_tet_mesh_report_dict(report)), map);
    }
    auto [tets, report] = tf::make_tet_mesh(form, config, tf::return_report);
    return lanes(tets, make_tet_mesh_report_dict(report));
  }
  if (return_index_map) {
    auto [tets, status, map] =
        tf::make_tet_mesh(form, config, tf::return_index_map);
    return with_map(lanes(tets, nanobind::int_(int(status))), map);
  }
  auto [tets, status] = tf::make_tet_mesh(form, config);
  return lanes(tets, nanobind::int_(int(status)));
}

} // namespace tf::py
