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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/classify.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "../../topology/edge_id_in_face.hpp"
#include "../../topology/topo_type.hpp"
#include "../loop/vertex_source.hpp"
#include "../impl/polygon_arrangement_labels.hpp"
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

template <typename Tup0, typename Policy0, typename Policy1, typename Policy2,
          typename Range, typename Zipped, typename LabelType, typename Index,
          typename LocalR>
auto classify_by_wedge_on_shared_edge(
    const Tup0 &tup0, std::size_t edge_id, const Range &neighbors,
    const Zipped &zipped, const tf::polygons<Policy0> &polygons0,
    const tf::polygons<Policy1> &polygons1,
    const tf::cut::polygon_arrangement_labels<LabelType> &pal1,
    Index partition_id, LocalR &local_r,
    const tf::points<Policy2> &intersection_points) {

  auto &&[loop0, mapped_loop0, d0] = tup0;
  auto next_edge_id =
      tf::circular_increment(edge_id, std::size_t{loop0.size()});

  // Edge direction vector (normalized)
  auto u = tf::normalized(intersection_points[mapped_loop0[next_edge_id].id] -
                          intersection_points[mapped_loop0[edge_id].id]);

  auto frame0 = tf::frame_of(polygons0);
  auto frame1 = tf::frame_of(polygons1);

  const auto &poly0 = polygons0[d0.object];
  auto plane0 = tf::transformed(tf::make_plane(poly0), frame0);

  using RealT = tf::coordinate_type<Policy2>;

  // Lambda: classify other-mesh faces using wedge (plane0 + wedge_plane)
  auto classify_with_wedge = [&, &d0 = d0, &loop0 = loop0](const auto &wedge_plane) {
    for (auto other_id : neighbors) {
      const auto &[loop_other, mapped_other, d_other] = zipped[other_id];
      if (d_other.tag == d0.tag)
        continue; // skip same mesh

      const auto &poly_other = polygons1[d_other.object];
      auto plane_other = tf::transformed(tf::make_plane(poly_other), frame1);

      // Find edge in loop_other and get direction from its perspective
      auto other_edge_id =
          tf::edge_id_in_face(loop0[edge_id], loop0[next_edge_id], loop_other);
      if (other_edge_id == loop_other.size())
        continue;
      auto other_next_edge_id =
          tf::circular_increment(other_edge_id, std::size_t{loop_other.size()});
      auto u_other = tf::normalized(intersection_points[mapped_other[other_next_edge_id].id] -
                                    intersection_points[mapped_other[other_edge_id].id]);

      // Vector in other_plane perpendicular to edge, pointing "into" the face
      auto v_other = tf::cross(plane_other.normal, u_other);

      // Compute vA and vB: vectors in planes A and B, perpendicular to edge, pointing into faces
      // A has edge direction u, B has edge direction -u
      auto vA = tf::cross(plane0.normal, u);
      auto vB = tf::cross(wedge_plane.normal, -u);

      // Check coplanarity with A and B
      auto d0_dot = tf::dot(plane0.normal, v_other);
      auto dw_dot = tf::dot(wedge_plane.normal, v_other);

      constexpr auto eps = tf::epsilon<RealT>;

      // Check if C is on A's or B's half-plane:
      // - Coplanar with the plane (d_dot ≈ 0)
      // - AND v_other points same direction as vA/vB
      bool coplanar_A = std::abs(d0_dot) < eps;
      bool coplanar_B = std::abs(dw_dot) < eps;
      bool on_A = coplanar_A && tf::dot(vA, v_other) > 0;
      bool on_B = coplanar_B && tf::dot(vB, v_other) > 0;

      // Edge direction in C compared to A's edge direction
      bool same_as_A = (loop_other[other_edge_id] == loop0[edge_id]);

      int classification;
      if (on_A) {
        // C is on A's half-plane
        // same_as_A → aligned, opposite → opposing
        classification = same_as_A ? 2 : 3;
      } else if (on_B) {
        // C is on B's half-plane
        // B has opposite edge direction to A, so:
        // same_as_A means opposite_to_B → opposing
        // opposite_to_A means same_as_B → aligned
        classification = same_as_A ? 3 : 2;
      } else if (d0_dot < 0 && dw_dot < 0) {
        classification = 0; // inside
      } else {
        classification = 1; // outside
      }

      auto label_other = pal1.cut_labels[other_id - partition_id];
      local_r[d_other.tag][label_other][classification]++;
    }
  };

  // Find wedge partner: same mesh, opposite edge direction
  for (auto n_id : neighbors) {
    const auto &[loop1, mapped_loop1, d1] = zipped[n_id];
    if (d1.tag != d0.tag)
      continue;

    auto other_edge_id =
        tf::edge_id_in_face(loop0[edge_id], loop0[next_edge_id], loop1);
    if (other_edge_id == loop1.size())
      continue;

    bool same_direction = loop0[edge_id] == loop1[other_edge_id];
    if (same_direction)
      continue; // need opposite direction

    // Found wedge partner
    const auto &poly1 = polygons0[d1.object];
    auto wedge_plane = tf::transformed(tf::make_plane(poly1), frame0);

    classify_with_wedge(wedge_plane);
  }
}


} // namespace tf::cut
