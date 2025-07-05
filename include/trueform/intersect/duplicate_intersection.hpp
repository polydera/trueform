/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/circular_increment.hpp"
#include "../topology/edge_id_in_face.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/manifold_edge_link.hpp"
#include "./intersection.hpp"

namespace tf::intersect {
template <typename Policy, typename Index, std::size_t N>
auto duplicate_intersection(
    const tf::faces<Policy> &faces,
    tf::intersect::intersection<Index> intersection,
    const tf::face_membership<Index> &fe,
    const tf::manifold_edge_link<Index, N> &mel,
    tf::buffer<tf::intersect::intersection<Index>> &intersections) {
  auto make_push = [&](tf::intersect::intersection<Index> intersection) {
    intersections.push_back(intersection);
    intersection.mesh = 1;
    std::swap(intersection.polygon, intersection.polygon_other);
    std::swap(intersection.target, intersection.target_other);
    intersections.push_back(intersection);
  };
  if (intersection.target_other.label == tf::topo_type::face) {
    make_push(intersection);
  } else if (intersection.target_other.label == tf::topo_type::vertex) {
    // all polygons containing the vertex will be processed
    Index pt_id =
        faces[intersection.polygon_other][intersection.target_other.id];
    for (auto poly_id : fe[pt_id]) {
      Index n_pt_id =
          std::find(faces[poly_id].begin(), faces[poly_id].end(), pt_id) -
          faces[poly_id].begin();
      auto n_intersection = intersection;
      n_intersection.polygon_other = poly_id;
      n_intersection.target_other.id = n_pt_id;
      make_push(n_intersection);
    }
  } else if (intersection.target_other.label == tf::topo_type::edge) {
    // we only process the neighbors further down
    make_push(intersection);
    Index e0 = faces[intersection.polygon_other][intersection.target_other.id];
    Index e1 = faces[intersection.polygon_other][tf::circular_increment<Index>(
        intersection.target_other.id, Index(N))];
    if (mel[intersection.polygon_other][intersection.target_other.id]
            .is_simple()) {
      Index n_poly_id =
          mel[intersection.polygon_other][intersection.target_other.id]
              .face_peer;
      Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
      auto n_intersection = intersection;
      n_intersection.polygon_other = n_poly_id;
      n_intersection.target_other.id = n_e;
      make_push(n_intersection);
    } else if (!mel[intersection.polygon_other][intersection.target_other.id]
                    .is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe, faces, intersection.polygon_other, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_poly_id : neighbors) {
        Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
        auto n_intersection = intersection;
        n_intersection.polygon_other = n_poly_id;
        n_intersection.target_other.id = n_e;
        make_push(n_intersection);
      }
    }
  }
}

template <typename Policy0, typename Policy1, typename Index, std::size_t N0,
          std::size_t N1>
auto duplicate_intersection(
    const tf::faces<Policy0> &faces0, const tf::faces<Policy1> &faces1,
    tf::intersect::intersection<Index> intersection,
    const tf::face_membership<Index> &fe0,
    const tf::manifold_edge_link<Index, N0> &mel0,
    const tf::face_membership<Index> &fe1,
    const tf::manifold_edge_link<Index, N1> &mel1,
    tf::buffer<tf::intersect::intersection<Index>> &intersections) {

  if (intersection.target.label == tf::topo_type::face) {
    duplicate_intersection(faces1, intersection, fe1, mel1, intersections);

  } else if (intersection.target.label == tf::topo_type::vertex) {
    // all polygons containing the vertex will be processed
    Index pt_id = faces0[intersection.polygon][intersection.target.id];
    for (auto poly_id : fe0[pt_id]) {
      Index n_pt_id =
          std::find(faces0[poly_id].begin(), faces0[poly_id].end(), pt_id) -
          faces0[poly_id].begin();
      auto n_intersection = intersection;
      n_intersection.polygon = poly_id;
      n_intersection.target.id = n_pt_id;
      duplicate_intersection(faces1, n_intersection, fe1, mel1, intersections);
    }
  } else if (intersection.target.label == tf::topo_type::edge) {
    // we only process the neighbors further down
    duplicate_intersection(faces1, intersection, fe1, mel1, intersections);
    Index e0 = faces0[intersection.polygon][intersection.target.id];
    Index e1 = faces0[intersection.polygon][tf::circular_increment<Index>(
        intersection.target.id, Index(N0))];
    if (mel0[intersection.polygon][intersection.target.id].is_simple()) {
      Index n_poly_id =
          mel0[intersection.polygon][intersection.target.id].face_peer;
      Index n_e = tf::edge_id_in_face(e1, e0, faces0[n_poly_id]);
      auto n_intersection = intersection;
      n_intersection.polygon = n_poly_id;
      n_intersection.target.id = n_e;
      duplicate_intersection(faces1, n_intersection, fe1, mel1, intersections);
    } else if (!mel0[intersection.polygon][intersection.target.id]
                    .is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe0, faces0, intersection.polygon, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_poly_id : neighbors) {
        Index n_e = tf::edge_id_in_face(e1, e0, faces0[n_poly_id]);
        auto n_intersection = intersection;
        n_intersection.polygon = n_poly_id;
        n_intersection.target.id = n_e;
        duplicate_intersection(faces1, n_intersection, fe1, mel1,
                               intersections);
      }
    }
  }
}

} // namespace tf::intersect
