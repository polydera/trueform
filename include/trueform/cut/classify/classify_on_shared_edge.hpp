/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/classify.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "../../topology/edge_id_in_face.hpp"
#include "../../topology/topo_type.hpp"
#include "../loop/vertex_source.hpp"
#include <cstddef>

namespace tf::cut {
template <typename Tup0, typename Tup1, typename Policy0, typename Policy1,
          typename Policy2>
auto classify_on_shared_edge(const Tup0 &tup0, std::size_t edge_id,
                             const Tup1 &tup1,
                             const tf::polygons<Policy0> &polygons0,
                             const tf::polygons<Policy1> &polygons1,
                             const tf::points<Policy2> &intersection_points) {
  auto &&[loop0, mapped_loop0, d0, counts0] = tup0;
  auto &&[loop1, mapped_loop1, d1, counts1] = tup1;
  auto next_edge_id =
      tf::circular_increment(edge_id, std::size_t{loop0.size()});
  if (loop0[edge_id] == loop0[next_edge_id])
    return;
  auto other_edge_id =
      tf::edge_id_in_face(loop0[edge_id], loop0[next_edge_id], loop1);
  if (other_edge_id == loop1.size())
    return;
  auto same_direction = loop0[edge_id] == loop1[other_edge_id];
  // already in world coordinates
  auto u = intersection_points[mapped_loop0[next_edge_id].id] -
           intersection_points[mapped_loop0[edge_id].id];
  auto frame0 = tf::frame_of(polygons0);
  auto frame1 = tf::frame_of(polygons1);
  const auto &poly0 = polygons0[d0.object];
  const auto &poly1 = polygons1[d1.object];
  auto plane0 = tf::transformed(tf::make_plane(poly0), frame0);
  auto plane1 = tf::transformed(tf::make_plane(poly1), frame1);
  bool out0 = tf::dot(plane1.normal, tf::cross(plane0.normal, u)) > 0;
  bool out1 = out0;
  if (same_direction)
    out1 = !out1;
  counts0[out0]++;
  counts1[out1]++;
}

template <typename Tup0, typename Tup1, typename Range, typename Policy0,
          typename Policy1>
auto classify_on_shared_edge(const Tup0 &tup0, std::size_t edge_id,
                             const Tup1 &tup1, const Range &intersections,
                             const tf::polygons<Policy0> &polygons0,
                             const tf::polygons<Policy1> &polygons1) {
  auto &&[loop0, mapped_loop0, d0, counts0] = tup0;
  auto &&[loop1, mapped_loop1, d1, counts1] = tup1;
  auto next_edge_id =
      tf::circular_increment(edge_id, std::size_t{loop0.size()});
  auto other_edge_id =
      tf::edge_id_in_face(loop0[edge_id], loop0[next_edge_id], loop1);
  if (other_edge_id == loop1.size())
    return;
  auto same_direction = loop0[edge_id] == loop1[other_edge_id];

  auto f = [same_direction, &intersections](
               const auto &mapped_loop0, const auto &d0, auto &counts0,
               const auto &d1, auto &counts1, auto edge_id,
               const auto &polygons0, const auto &polygons1) {
    auto v0 = mapped_loop0[edge_id];
    if (v0.source == loop::vertex_source::original)
      return false;
    auto ins = intersections[v0.intersection_index];
    if (ins.target.label == tf::topo_type::face)
      return false;
    auto frame0 = tf::frame_of(polygons0);
    auto frame1 = tf::frame_of(polygons1);
    const auto &poly0 = polygons0[d0.object];
    const auto &poly1 = polygons1[d1.object];
    auto plane1 = tf::transformed(tf::make_plane(poly1), frame1);
    auto pt0 = tf::transformed(poly0[ins.target.id], frame0);
    auto test = tf::classify(pt0, plane1) == tf::sidedness::on_positive_side;
    counts0[test]++;
    if (same_direction)
      test = !test;
    counts1[test]++;
    return true;
  };
  if (!f(mapped_loop0, d0, counts0, d1, counts1, edge_id, polygons0, polygons1))
    f(mapped_loop1, d1, counts1, d0, counts0, other_edge_id, polygons1,
      polygons0);
}

} // namespace tf::cut
