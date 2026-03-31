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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"
#include "../../spatial/classify/point_in_mesh.hpp"
#include "../../spatial/signed_distance.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"
#include "../partition/make_set_component_labels.hpp"
#include "../partition/partition_labels.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {

/// Classify components that have no intersection edges (zero wedge votes)
/// using signed distance against the other mesh.
///
/// For each component with all-zero counts, picks a representative
/// point (centroid of a polygon or cut face loop) and queries
/// signed_distance against the other mesh.
template <typename Index, typename LabelType, typename Policy0,
          typename Policy1, typename Int, typename GetCreatedPoint>
void classify_missing_by_signed_distance(
    tf::blocked_buffer<Index, 4> &counts, const tf::polygons<Policy0> &polygons,
    const tf::polygons<Policy1> &polygons_other,
    const tf::cut::partition_labels<LabelType> &pal,
    const tf::face_cuts<Index, Int> &fc, std::size_t tag_self,
    const GetCreatedPoint &get_created_point) {
  using vertex_t = intersect::graph::vertex<Index>;
  using source = intersect::graph::vertex_source;
  using real_type = tf::coordinate_type<Policy0>;

  bool any = false;
  for (auto &&c : counts)
    if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) {
      any = true;
      break;
    }
  if (!any)
    return;

  // We collect a representative for each component.
  // Technically this is a race condition. But we
  // do not care which id gets written,
  // and parallel_for_each end synchronizes.
  auto make_reprs = [](const auto &labels, auto n_components) {
    tf::buffer<Index> reprs;
    reprs.allocate(n_components);
    tf::parallel_fill(reprs, Index(-1));
    tf::parallel_for_each(tf::enumerate(labels), [&](auto pair) {
      const auto &[id, label] = pair;
      if (label == -1)
        return;
      reprs[label] = id;
    });
    return reprs;
  };

  auto reprs_poly = make_reprs(pal.polygon_labels, pal.n_components);
  auto reprs_cut = make_reprs(pal.cut_labels, pal.n_components);

  auto frame = tf::frame_of(polygons);
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());
  auto loops = loops_per_tag[tag_self];

  auto get_loop_point = [&](const vertex_t &v) {
    constexpr auto dims = tf::coordinate_dims_v<Policy0>;
    tf::point<real_type, dims> pt;
    if (v.source == source::original)
      pt = tf::transformed(polygons.points()[v.id], frame);
    else
      pt = get_created_point(v.id);
    return pt;
  };

  tf::parallel_for_each(tf::zip(counts, reprs_poly, reprs_cut), [&](auto tup) {
    auto &&[count, poly_id, cut_id] = tup;
    if (!(count[0] == 0 && count[1] == 0 && count[2] == 0 && count[3] == 0))
      return;
    constexpr auto dims = tf::coordinate_dims_v<Policy0>;
    tf::point<real_type, dims> point;
    if (poly_id != Index(-1))
      point = tf::transformed(tf::centroid(polygons[poly_id]), frame);
    else {
      auto loop = loops[cut_id];
      point = tf::centroid(
          tf::make_polygon(tf::make_mapped_range(loop, get_loop_point)));
    }
    auto sd = tf::signed_distance(polygons_other, point);
    bool is_outside = sd > real_type(0);
    count[0] = 0;
    count[1] = 0;
    count[2] = 0;
    count[3] = 0;
    count[is_outside] = 1;
  });
}

/// Classify components that have no intersection edges (zero wedge votes)
/// using ray-based containment through joint components.
///
/// Builds set_component_labels for the other mesh (merging cut face
/// components connected through same-tag manifold edges), then uses
/// classify_point for each missing component.
template <typename Index, typename LabelType, typename Policy0,
          typename Policy1, typename Int, typename GetCreatedPoint>
void classify_missing_by_containment(
    tf::blocked_buffer<Index, 4> &counts, const tf::polygons<Policy0> &polygons,
    const tf::polygons<Policy1> &polygons_other,
    const tf::cut::partition_labels<LabelType> &pal,
    const tf::cut::partition_labels<LabelType> &pal_other,
    const tf::face_cuts<Index, Int> &fc, const tf::cut_graph<Index> &cg,
    std::size_t tag_self, std::size_t tag_other,
    const GetCreatedPoint &get_created_point) {
  using vertex_t = intersect::graph::vertex<Index>;
  using source = intersect::graph::vertex_source;
  using real_type = tf::coordinate_type<Policy0>;

  bool any = false;
  for (auto &&c : counts)
    if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) {
      any = true;
      break;
    }
  if (!any)
    return;

  auto descs = fc.descriptors();
  auto tag_offsets = fc.tag_offsets();
  auto conn = cg.connectivity_per_face_edge(fc);
  auto descs_per_tag = tf::make_offset_block_range(tag_offsets, descs);
  auto conn_per_tag = tf::make_offset_block_range(fc.tag_offsets(), conn);
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  auto jlabels_other = tf::cut::make_set_component_labels(
      polygons_other, descs_per_tag[tag_other], conn_per_tag[tag_other], descs,
      pal_other, tag_offsets[tag_other]);

  // We collect a representative for each component.
  // Technically this is a race condition. But we
  // do not care which id gets written,
  // and parallel_for_each end synchronizes.
  auto make_reprs = [](const auto &labels, auto n_components) {
    tf::buffer<Index> reprs;
    reprs.allocate(n_components);
    tf::parallel_fill(reprs, Index(-1));
    tf::parallel_for_each(tf::enumerate(labels), [&](auto pair) {
      const auto &[id, label] = pair;
      if (label == -1)
        return;
      reprs[label] = id;
    });
    return reprs;
  };

  auto reprs_poly = make_reprs(pal.polygon_labels, pal.n_components);
  auto reprs_cut = make_reprs(pal.cut_labels, pal.n_components);

  auto frame = tf::frame_of(polygons);
  auto loops = loops_per_tag[tag_self];

  auto get_loop_point = [&](const vertex_t &v) {
    constexpr auto dims = tf::coordinate_dims_v<Policy0>;
    tf::point<real_type, dims> pt;
    if (v.source == source::original)
      pt = tf::transformed(polygons.points()[v.id], frame);
    else
      pt = get_created_point(v.id);
    return pt;
  };

  tf::parallel_for_each(tf::zip(counts, reprs_poly, reprs_cut), [&](auto tup) {
    auto &&[count, poly_id, cut_id] = tup;
    if (!(count[0] == 0 && count[1] == 0 && count[2] == 0 && count[3] == 0))
      return;
    constexpr auto dims = tf::coordinate_dims_v<Policy0>;
    tf::point<real_type, dims> point;
    if (poly_id != Index(-1))
      point = tf::transformed(tf::centroid(polygons[poly_id]), frame);
    else {
      auto loop = loops[cut_id];
      point = tf::centroid(
          tf::make_polygon(tf::make_mapped_range(loop, get_loop_point)));
    }
    auto c = tf::spatial::classify_point(point, polygons_other, jlabels_other);
    count[c == tf::containment::outside] = 1;
  });
}

/// Dispatch missing component classification for a 2-mesh pair.
///
/// Runs both meshes in parallel. Uses containment (ray cast through
/// joint components) when use_containment is true, otherwise falls
/// back to signed distance.
template <typename Index, typename LabelType, typename Policy0,
          typename Policy1, typename Int, typename GetCreatedPoint>
void classify_missing(tf::blocked_buffer<Index, 4> &counts0,
                      tf::blocked_buffer<Index, 4> &counts1,
                      const tf::polygons<Policy0> &polygons0,
                      const tf::polygons<Policy1> &polygons1,
                      const tf::cut::partition_labels<LabelType> &pal0,
                      const tf::cut::partition_labels<LabelType> &pal1,
                      const tf::face_cuts<Index, Int> &fc,
                      const tf::cut_graph<Index> &cg, bool use_containment,
                      const GetCreatedPoint &get_created_point) {
  if (use_containment) {
    tbb::parallel_invoke(
        [&] {
          classify_missing_by_containment(counts0, polygons0, polygons1, pal0,
                                          pal1, fc, cg, 0, 1,
                                          get_created_point);
        },
        [&] {
          classify_missing_by_containment(counts1, polygons1, polygons0, pal1,
                                          pal0, fc, cg, 1, 0,
                                          get_created_point);
        });
  } else {
    tbb::parallel_invoke(
        [&] {
          classify_missing_by_signed_distance(counts0, polygons0, polygons1,
                                              pal0, fc, 0, get_created_point);
        },
        [&] {
          classify_missing_by_signed_distance(counts1, polygons1, polygons0,
                                              pal1, fc, 1, get_created_point);
        });
  }
}

} // namespace tf::cut
