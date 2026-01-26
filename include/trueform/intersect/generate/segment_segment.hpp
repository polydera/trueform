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
#include "../../core/buffer.hpp"
#include "../../core/intersects.hpp"
#include "../../core/ray.hpp"
#include "../../core/ray_hit.hpp"
#include "../../topology/edge_membership_like.hpp"
#include "../types/intersection.hpp"

namespace tf::intersect::generate {
namespace detail {
template <typename F, typename Policy, typename Index, typename RealT>
auto segment_segment(const F &is_representative,
                     const tf::segment<2, Policy> &seg0,
                     const tf::segment<2, Policy> &seg1,
                     tf::buffer<intersect::intersection<Index>> &intersections,
                     tf::buffer<tf::point<RealT, 2>> &points) {
  using real_t = tf::coordinate_type<Policy>;
  auto generate_vertex_edge = [&intersections, &points](
                                  auto r0, auto v_id, auto edge_id0, auto v_pt,
                                  auto edge_id, const auto &segment) {
    if (r0 && v_id != segment.indices()[0] && v_id != segment.indices()[1]) {
      auto ray = tf::make_ray_between_points(segment[0], segment[1]);
      auto hit =
          tf::ray_hit(ray, v_pt, tf::make_ray_config(real_t(0), real_t(1)));
      if (hit) {
        Index pt_id = points.size();
        points.push_back(v_pt);
        intersections.push_back({Index(edge_id0),
                                 Index(edge_id),
                                 {Index(v_id), tf::topo_type::vertex},
                                 {Index(edge_id), tf::topo_type::edge},
                                 pt_id});
      }
    }
  };

  auto generate_vertex_vertex = [&intersections,
                                 &points](auto r0, auto v_id0, auto edge_id0,
                                          auto v_pt0, auto r1, auto v_id1,
                                          auto edge_id1, auto v_pt1) {
    if (r0 && r1 && v_id0 != v_id1) {
      if (tf::intersects(v_pt0, v_pt1)) {
        Index pt_id = points.size();
        points.push_back(v_pt0);
        intersections.push_back({Index(edge_id0),
                                 Index(edge_id1),
                                 {Index(v_id0), tf::topo_type::vertex},
                                 {Index(v_id1), tf::topo_type::vertex},
                                 pt_id});
      }
    }
  };

  auto generate_edge_edge = [&intersections,
                             &points](auto edge_id0, const auto &segment0,
                                      auto edge_id1, const auto &segment1) {
    if (segment0.indices()[0] == segment1.indices()[0] ||
        segment0.indices()[1] == segment1.indices()[0] ||
        segment0.indices()[0] == segment1.indices()[1] ||
        segment0.indices()[1] == segment1.indices()[1])
      return;
    auto ray = tf::make_ray_between_points(segment0[0], segment0[1]);
    auto hit =
        tf::ray_hit(ray, segment1, tf::make_ray_config(real_t(0), real_t(1)));
    if (hit) {
      Index pt_id = points.size();
      points.push_back(hit.point);
      intersections.push_back({Index(edge_id0),
                               Index(edge_id1),
                               {Index(edge_id0), tf::topo_type::edge},
                               {Index(edge_id1), tf::topo_type::edge},
                               pt_id});
    }
  };
  auto id0 = seg0.id();
  auto id1 = seg1.id();
  for (int i = 0; i < 2; ++i) {
    auto vid0 = seg0.indices()[i];
    auto r0 = is_representative(vid0, id0);
    auto pt0 = seg0[i];
    generate_vertex_edge(r0, vid0, id0, pt0, id1, seg1);
    for (int j = 0; j < 2; ++j) {
      auto vid1 = seg1.indices()[j];
      auto r1 = is_representative(vid1, id1);
      auto pt1 = seg1[j];
      generate_vertex_vertex(r0, vid0, id0, pt0, r1, vid1, id1, pt1);
    }
  }
  for (int i = 0; i < 2; ++i) {
    auto vid1 = seg1.indices()[i];
    auto r1 = is_representative(vid1, id1);
    auto pt1 = seg1[i];
    generate_vertex_edge(r1, vid1, id1, pt1, id0, seg0);
  }
  generate_edge_edge(id0, seg0, id1, seg1);
}
} // namespace detail

template <typename Policy0, typename Policy, typename Index, typename RealT>
auto segment_segment(const tf::edge_membership_like<Policy0> &em,
                     const tf::segment<2, Policy> &seg0,
                     const tf::segment<2, Policy> &seg1,
                     tf::buffer<intersect::intersection<Index>> &intersections,
                     tf::buffer<tf::point<RealT, 2>> &points) {
  return detail::segment_segment(
      [&](auto vid, auto eid) { return em[vid].front() == eid; }, seg0, seg1,
      intersections, points);
}

template <typename Policy, typename Index, typename RealT>
auto segment_segment(const tf::segment<2, Policy> &seg0,
                     const tf::segment<2, Policy> &seg1,
                     tf::buffer<intersect::intersection<Index>> &intersections,
                     tf::buffer<tf::point<RealT, 2>> &points) {
  return detail::segment_segment([](auto, auto) { return true; }, seg0, seg1,
                                 intersections, points);
}
} // namespace tf::intersect::generate
