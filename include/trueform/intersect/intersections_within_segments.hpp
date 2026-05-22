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
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/intersects.hpp"
#include "../core/local_buffer.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../exact/projection_axes.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/segment_intersect.hpp"
#include "../spatial/search_self.hpp"
#include "../topology/policy/edge_membership.hpp"
#include "./exact/dedup_segment_vertex_points.hpp"
#include "./exact/duplicate_intersection.hpp"
#include "./exact/intersection.hpp"
#include "tbb/parallel_sort.h"

namespace tf {

/// @ingroup intersect_data
/// @brief Exact segment intersection data.
///
/// Computes all intersection points between segments using exact
/// int32 predicates. Supports 2D and 3D segments. For 3D, projection
/// axes are computed per pair via the dominant normal plane.
/// Points are stored as int32 internally and deconverted to float
/// for output.
template <typename Index, typename RealType, std::size_t Dims,
          typename Int = tf::exact::int32>
class intersections_within_segments {
  static_assert(Dims == 2 || Dims == 3, "Only 2D and 3D segments supported");

public:
  auto converter() const
      -> const tf::exact_coordinate_converter<Int, RealType, Dims> & {
    return _converter;
  }

  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersection_points() const {
    return tf::make_points(tf::make_range(_intersection_points));
  }

  auto clear() {
    _intersection_points.clear();
    _intersections_offsets.clear();
    _intersections.clear();
  }

  template <typename Policy> auto build(const tf::segments<Policy> &segments) {
    static_assert(tf::has_tree_policy<Policy>, "Use: segments | tf::tag(tree)");
    static_assert(tf::has_edge_membership_policy<Policy>,
                  "Use: segments | tf::tag(edge_membership)");
    static_assert(tf::coordinate_dims_v<Policy> == Dims,
                  "Dimension mismatch between segments and class template");
    clear();
    _converter = tf::make_exact_coordinate_converter<Int, RealType>(segments);

    auto [raw_intersections, raw_points] = generate_intersections(segments);

    if (raw_points.size() == 0) {
      finalize(Index(0));
      return;
    }

    // Deduplicate vertex points (VV merge + shared vertex collapse)
    tf::intersect::dedup_segment_vertex_points(
        raw_intersections, raw_points,
        [&](Index object, Index local_id) -> Index {
          return Index(segments.edges()[object][local_id]);
        });

    // Duplicate intersections across all incident edges at vertices
    auto &&em = segments.edge_membership();
    auto edges = segments.edges();
    tf::buffer<tf::intersect::intersection<Index>> expanded;
    if (Index(raw_intersections.size()) > 1000)
      tf::generic_generate(
          raw_intersections, expanded, [&](const auto &rec, auto &buffer) {
            tf::intersect::duplicate_intersection(rec, em, edges, buffer);
          });
    else {
      expanded.reserve(raw_intersections.size() * 2);
      for (const auto &rec : raw_intersections)
        tf::intersect::duplicate_intersection(rec, em, edges, expanded);
    }

    _intersection_points = std::move(raw_points);
    _intersections = std::move(expanded);
    finalize(Index(_intersection_points.size()));
  }

private:
  auto finalize(Index n_ids) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(
        _intersections, std::back_inserter(_intersections_offsets), Index(0),
        [](const auto &x0, const auto &x1) { return x0.object == x1.object; });
  }

  template <typename Policy>
  auto generate_intersections(const tf::segments<Policy> &segments) {
    if (segments.size() < 1000)
      return generate_intersections_impl(segments, 0);
    else
      return generate_intersections_impl(segments, 6);
  }

  template <typename Policy>
  auto generate_intersections_impl(const tf::segments<Policy> &segments,
                                   int parallelism_depth) {
    tf::local_buffer<tf::intersect::intersection<Index>> l_intersections;
    tf::local_buffer<tf::point<Int, Dims>> l_points;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    auto &conv = _converter;

    tf::search_self(
        segments,
        [&](const auto &bv0, const auto &bv1) {
          return tf::intersects(tf::make_aabb(conv(bv0.min), conv(bv0.max)),
                                tf::make_aabb(conv(bv1.min), conv(bv1.max)));
        },
        [&](const auto &seg0, const auto &seg1) {
          intersect_pair(seg0, seg1, segments.edge_membership(),
                         *l_intersections, *l_points);
        },
        parallelism_depth);

    tf::buffer<tf::intersect::intersection<Index>> intersections;
    tf::buffer<tf::point<Int, Dims>> points;
    l_points.to_buffer(points);
    intersections.allocate(l_intersections.total_size());
    auto it = intersections.begin();
    std::size_t offset = 0;
    for (const auto &v : l_intersections.buffers()) {
      for (auto e : v) {
        e.id += offset;
        *it++ = e;
      }
      offset += v.size();
    }
    return std::make_pair(std::move(intersections), std::move(points));
  }

  static auto get_axes(const tf::exact::vertex<Index, Int> &a0,
                       const tf::exact::vertex<Index, Int> &a1,
                       const tf::exact::vertex<Index, Int> &b0,
                       const tf::exact::vertex<Index, Int> &b1)
      -> std::pair<int, int> {
    if constexpr (Dims == 2)
      return {0, 1};
    else {
      auto pts_equal = [](const tf::exact::pt3<Int> &x,
                          const tf::exact::pt3<Int> &y) {
        return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
      };
      // Pick a third point that differs from a0 and a1
      if (!pts_equal(b0.pt, a0.pt) && !pts_equal(b0.pt, a1.pt))
        return tf::exact::projection_axes(a0.pt, a1.pt, b0.pt);
      return tf::exact::projection_axes(a0.pt, a1.pt, b1.pt);
    }
  }

  template <typename Point>
  auto make_vertex(Index id, const Point &p) const
      -> tf::exact::vertex<Index, Int> {
    auto converted = _converter(p);
    if constexpr (Dims == 2)
      return {id, tf::exact::pt3<Int>{converted[0], converted[1], 0}};
    else
      return {id, converted};
  }

  static auto make_point(const tf::exact::pt3<Int> &pt3, int a0, int a1)
      -> tf::point<Int, Dims> {
    if constexpr (Dims == 2)
      return {pt3[a0], pt3[a1]};
    else
      return {pt3[0], pt3[1], pt3[2]};
  }

  template <typename Seg0, typename Seg1, typename EM>
  auto intersect_pair(const Seg0 &seg0, const Seg1 &seg1, const EM &em,
                      tf::buffer<tf::intersect::intersection<Index>> &ints,
                      tf::buffer<tf::point<Int, Dims>> &pts) const {
    auto id0 = Index(seg0.id());
    auto id1 = Index(seg1.id());

    auto vid0_0 = Index(seg0.indices()[0]);
    auto vid0_1 = Index(seg0.indices()[1]);
    auto vid1_0 = Index(seg1.indices()[0]);
    auto vid1_1 = Index(seg1.indices()[1]);

    auto a0 = make_vertex(vid0_0, seg0[0]);
    auto a1 = make_vertex(vid0_1, seg0[1]);
    auto b0 = make_vertex(vid1_0, seg1[0]);
    auto b1 = make_vertex(vid1_1, seg1[1]);

    auto [ax0, ax1] = get_axes(a0, a1, b0, b1);

    auto result = tf::exact::classify_segments(a0, a1, b0, b1, ax0, ax1);
    if (!result)
      return;

    auto is_rep = [&](Index vid, Index eid) {
      return Index(em[vid].front()) == eid;
    };

    auto should_emit = [&](const tf::exact::segment_intersection<short> &hit) {
      bool a_is_vertex = hit.target_a.label == tf::topo_type::vertex;
      bool b_is_vertex = hit.target_b.label == tf::topo_type::vertex;
      // VV: both vertices must be representative on their edge
      if (a_is_vertex && b_is_vertex) {
        auto va = hit.target_a.id == 0 ? vid0_0 : vid0_1;
        auto vb = hit.target_b.id == 0 ? vid1_0 : vid1_1;
        return is_rep(va, id0) && is_rep(vb, id1);
      }
      // VE: the vertex must be representative on its edge
      if (a_is_vertex)
        return is_rep(hit.target_a.id == 0 ? vid0_0 : vid0_1, id0);
      if (b_is_vertex)
        return is_rep(hit.target_b.id == 0 ? vid1_0 : vid1_1, id1);
      // EE: always emit
      return true;
    };

    auto emit = [&, ax0 = ax0,
                 ax1 = ax1](const tf::exact::segment_intersection<short> &hit) {
      if (!should_emit(hit))
        return;
      auto pt = compute_point(hit, a0, a1, b0, b1, ax0, ax1);
      Index pt_id = pts.size();
      pts.push_back(pt);

      auto target = tf::topo_id<Index>{
          hit.target_a.label == tf::topo_type::vertex ? Index(hit.target_a.id)
                                                      : Index(id0),
          hit.target_a.label};
      auto target_other = tf::topo_id<Index>{
          hit.target_b.label == tf::topo_type::vertex ? Index(hit.target_b.id)
                                                      : Index(id1),
          hit.target_b.label};

      ints.push_back({Index(id0), Index(id1), target, target_other, pt_id});
    };

    emit(result->first);
    if (result->second)
      emit(*result->second);
  }

  static auto compute_point(const tf::exact::segment_intersection<short> &hit,
                            const tf::exact::vertex<Index, Int> &a0,
                            const tf::exact::vertex<Index, Int> &a1,
                            const tf::exact::vertex<Index, Int> &b0,
                            const tf::exact::vertex<Index, Int> &b1, int ax0,
                            int ax1) -> tf::point<Int, Dims> {
    if (hit.target_a.label == tf::topo_type::vertex) {
      auto &v = (hit.target_a.id == 0) ? a0 : a1;
      return make_point(v.pt, ax0, ax1);
    }
    if (hit.target_b.label == tf::topo_type::vertex) {
      auto &v = (hit.target_b.id == 0) ? b0 : b1;
      return make_point(v.pt, ax0, ax1);
    }
    if constexpr (Dims == 2) {
      tf::exact::pt2<Int> pa0 = {a0.pt[ax0], a0.pt[ax1]};
      tf::exact::pt2<Int> pa1 = {a1.pt[ax0], a1.pt[ax1]};
      tf::exact::pt2<Int> pb0 = {b0.pt[ax0], b0.pt[ax1]};
      tf::exact::pt2<Int> pb1 = {b1.pt[ax0], b1.pt[ax1]};
      return tf::exact::coplanar_edge_edge_point(pa0, pa1, pb0, pb1);
    } else {
      return tf::exact::coplanar_edge_edge_point(a0, a1, b0, b1, ax0, ax1);
    }
  }

  tf::exact_coordinate_converter<Int, RealType, Dims> _converter;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<tf::intersect::intersection<Index>> _intersections;
  tf::buffer<tf::point<Int, Dims>> _intersection_points;
};

} // namespace tf
