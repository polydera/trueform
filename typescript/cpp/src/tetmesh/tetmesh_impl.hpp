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

#include "trueform/core/blocked_buffer.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/core/return_report.hpp"
#include "trueform/core/tetrahedra_buffer.hpp"
#include "trueform/core/transformation.hpp"
#include "trueform/reindex/return_index_map.hpp"
#include "trueform/tetmesh/make_tet_mesh.hpp"
#include "trueform/tetmesh/tet_mesh_config.hpp"
#include "trueform/tetmesh/tet_mesh_index_map.hpp"
#include "trueform/tetmesh/tet_mesh_report.hpp"
#include "trueform/tetmesh/tet_mesh_status.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf {
namespace ts {

/// The recovery's face routes, as their counts.
struct tet_face_census_t {
  double covered, passes, flipped, points, unplaced, on_piece, refused,
      open_rim, crossed, off_lattice, budget, pierced, standing;
};

/// The constraint recovery's census.
struct tet_constraint_census_t {
  double segment_rounds, shell_splits, segments_refused, unrestored_facets;
  tet_face_census_t faces;
  double flip_recovered, search_recovered, split_rows, hull_barred;
  bool restored_before_faces;
  double retired_crossing, retired_neighbour, revived_segments, collapsed,
      collapse_refused, stalled_drains, budget_drains;
};

/// A tet mesh report; counts cross as numbers.
struct tet_mesh_report_t {
  int status;
  double refused_segments, refused_depth, refused_budget, refused_faces, rung,
      drains, pairs_named, pairs_cut, triangles_cut, dropped_degenerate,
      stitched_triangles, failed_triangles, pairs_unresolved,
      input_open_edges, recovered_open_edges;
  int storage_bits, input_bits, frame_bits;
  double deep_sites;
  int deepest, corner_bits_needed;
  double min_angle, flat, inverted, negative_cells;
  tet_constraint_census_t census;
};

/// A tet mesh: [T, 4] corners into [P, 3] double points, the status, and
/// the report, measured and filled only when it was asked for.
struct tet_mesh_t {
  wasm_ndarray<int> tetrahedra;
  wasm_ndarray<double> points;
  int status;
  tet_mesh_report_t report;
};

/// A tet mesh with its index map: the facet triangles as [K + 1] offsets
/// beside [N, 3] rows, and the face facets as [F + 1] offsets beside data.
struct tet_mesh_with_index_map_t {
  wasm_ndarray<int> tetrahedra;
  wasm_ndarray<double> points;
  int status;
  tet_mesh_report_t report;
  wasm_ndarray<int> facet_offsets;
  wasm_ndarray<int> facet_triangles;
  wasm_ndarray<int> face_facet_offsets;
  wasm_ndarray<int> face_facets;
};

inline auto make_tet_mesh_report(const tf::tet_mesh_report &r)
    -> tet_mesh_report_t {
  const auto &c = r.census;
  const auto &f = c.faces;
  return {int(r.status),
          double(r.refused_segments),
          double(r.refused_depth),
          double(r.refused_budget),
          double(r.refused_faces),
          double(r.rung),
          double(r.drains),
          double(r.pairs_named),
          double(r.pairs_cut),
          double(r.triangles_cut),
          double(r.dropped_degenerate),
          double(r.stitched_triangles),
          double(r.failed_triangles),
          double(r.pairs_unresolved),
          double(r.input_open_edges),
          double(r.recovered_open_edges),
          r.storage_bits,
          r.input_bits,
          r.frame_bits,
          double(r.deep_sites),
          r.deepest,
          r.corner_bits_needed,
          r.min_angle,
          double(r.flat),
          double(r.inverted),
          double(r.negative_cells),
          {double(c.segment_rounds),
           double(c.shell_splits),
           double(c.segments_refused),
           double(c.unrestored_facets),
           {double(f.covered), double(f.passes), double(f.flipped),
            double(f.points), double(f.unplaced), double(f.on_piece),
            double(f.refused), double(f.open_rim), double(f.crossed),
            double(f.off_lattice), double(f.budget), double(f.pierced),
            double(f.standing)},
           double(c.flip_recovered),
           double(c.search_recovered),
           double(c.split_rows),
           double(c.hull_barred),
           c.restored_before_faces,
           double(c.retired_crossing),
           double(c.retired_neighbour),
           double(c.revived_segments),
           double(c.collapsed),
           double(c.collapse_refused),
           double(c.stalled_drains),
           double(c.budget_drains)}};
}

/// The tetrahedra and their points as WASM lanes.
inline auto make_tet_mesh_lanes(tf::tetrahedra_buffer<int, double> &&tets)
    -> std::pair<wasm_ndarray<int>, wasm_ndarray<double>> {
  const int n_tets = int(tets.tetras_buffer().size());
  const int n_points = int(tets.points_buffer().size());
  return {wasm_ndarray<int>::from_buffer(
              std::move(tets.tetras_buffer().data_buffer()), {n_tets, 4}),
          wasm_ndarray<double>::from_buffer(
              std::move(tets.points_buffer().data_buffer()), {n_points, 3})};
}

/// `fn` on the mesh as a form, its face membership, manifold edge link and
/// transformation tagged, and on the config the fields state.
template <typename Real, typename Fn>
auto tet_mesh_call(wasm_mesh<Real> &m, bool split_constraints, double tolerance,
                   bool has_budget, int site_budget_ratio, Fn &&fn) {
  const auto pose = m.has_transformation()
                        ? tf::transformation<Real, 3>(m.transformation_view())
                        : tf::make_identity_transformation<Real, 3>();
  const auto form = m.polygons_range() | tf::tag(m.face_membership_range()) |
                    tf::tag(m.manifold_edge_link_range()) | tf::tag(pose);
  tf::tet_mesh_config config{split_constraints, tolerance};
  if (has_budget)
    config.site_budget_ratio = std::size_t(site_budget_ratio);
  return fn(form, config);
}

/// The status and, where it was asked for, the report of one call's answer.
inline auto make_tet_mesh_answer(tf::tet_mesh_status status)
    -> std::pair<int, tet_mesh_report_t> {
  return {int(status), tet_mesh_report_t{}};
}

inline auto make_tet_mesh_answer(const tf::tet_mesh_report &report)
    -> std::pair<int, tet_mesh_report_t> {
  return {int(report.status), make_tet_mesh_report(report)};
}

template <typename Real>
auto sync_tet_mesh(wasm_mesh<Real> &m, bool split_constraints, double tolerance,
                   bool has_budget, int site_budget_ratio, bool return_report)
    -> tet_mesh_t {
  return tet_mesh_call(
      m, split_constraints, tolerance, has_budget, site_budget_ratio,
      [return_report](const auto &form, const auto &config) -> tet_mesh_t {
        const auto emit = [](auto &&result) -> tet_mesh_t {
          auto lanes = make_tet_mesh_lanes(std::move(std::get<0>(result)));
          auto answer = make_tet_mesh_answer(std::get<1>(result));
          return {std::move(lanes.first), std::move(lanes.second),
                  answer.first, answer.second};
        };
        if (return_report)
          return emit(tf::make_tet_mesh(form, config, tf::return_report));
        return emit(tf::make_tet_mesh(form, config));
      });
}

template <typename Real>
auto sync_tet_mesh_with_index_map(wasm_mesh<Real> &m, bool split_constraints,
                                  double tolerance, bool has_budget,
                                  int site_budget_ratio, bool return_report)
    -> tet_mesh_with_index_map_t {
  return tet_mesh_call(
      m, split_constraints, tolerance, has_budget, site_budget_ratio,
      [return_report](const auto &form,
                      const auto &config) -> tet_mesh_with_index_map_t {
        const auto emit = [](auto &&result) -> tet_mesh_with_index_map_t {
          auto lanes = make_tet_mesh_lanes(std::move(std::get<0>(result)));
          auto answer = make_tet_mesh_answer(std::get<1>(result));
          auto &map = std::get<2>(result);
          const auto &rows = map.facet_triangles.data_buffer();
          tf::buffer<int> flat;
          flat.allocate(rows.size() * 3);
          for (std::size_t k = 0; k < rows.size(); ++k)
            for (std::size_t c = 0; c < 3; ++c)
              flat[k * 3 + c] = rows[k][c];
          const int n_rows = int(rows.size());
          return {std::move(lanes.first),
                  std::move(lanes.second),
                  answer.first,
                  answer.second,
                  wasm_ndarray<int>::from_buffer(
                      std::move(map.facet_triangles.offsets_buffer())),
                  wasm_ndarray<int>::from_buffer(std::move(flat), {n_rows, 3}),
                  wasm_ndarray<int>::from_buffer(
                      std::move(map.face_facets.offsets_buffer())),
                  wasm_ndarray<int>::from_buffer(
                      std::move(map.face_facets.data_buffer()))};
        };
        if (return_report)
          return emit(tf::make_tet_mesh(form, config, tf::return_report,
                                        tf::return_index_map));
        return emit(tf::make_tet_mesh(form, config, tf::return_index_map));
      });
}

template <typename Real>
auto async_tet_mesh(wasm_mesh<Real> &m, bool split_constraints, double tolerance,
                    bool has_budget, int site_budget_ratio, bool return_report)
    -> promise_t {
  return promise([a = m, split_constraints, tolerance, has_budget, site_budget_ratio,
                  return_report]() -> tet_mesh_t {
    return sync_tet_mesh<Real>(const_cast<wasm_mesh<Real> &>(a), split_constraints,
                               tolerance, has_budget, site_budget_ratio,
                               return_report);
  });
}

template <typename Real>
auto async_tet_mesh_with_index_map(wasm_mesh<Real> &m, bool split_constraints,
                                   double tolerance, bool has_budget,
                                   int site_budget_ratio, bool return_report)
    -> promise_t {
  return promise([a = m, split_constraints, tolerance, has_budget, site_budget_ratio,
                  return_report]() -> tet_mesh_with_index_map_t {
    return sync_tet_mesh_with_index_map<Real>(
        const_cast<wasm_mesh<Real> &>(a), split_constraints, tolerance, has_budget,
        site_budget_ratio, return_report);
  });
}

} // namespace ts
} // namespace tf
