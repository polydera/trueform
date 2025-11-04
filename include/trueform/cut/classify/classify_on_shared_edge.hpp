/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "../../topology/edge_id_in_face.hpp"
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
  if(loop0[edge_id] == loop0[next_edge_id])
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
} // namespace tf::cut
