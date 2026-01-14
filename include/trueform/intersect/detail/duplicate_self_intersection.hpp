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
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/edge_id_in_face.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../../topology/manifold_edge_link_like.hpp"
#include "../types/intersection.hpp"

namespace tf::intersect {
template <typename Policy, typename Index, typename Policy1, typename Policy2>
auto duplicate_self_intersection2(
    const tf::faces<Policy> &faces,
    tf::intersect::intersection<Index> intersection,
    const tf::face_membership_like<Policy1> &fe,
    const tf::manifold_edge_link_like<Policy2> &mel,
    tf::buffer<tf::intersect::intersection<Index>> &intersections) {
  auto make_push = [&](tf::intersect::intersection<Index> intersection) {
    intersections.push_back(intersection);
    std::swap(intersection.object, intersection.object_other);
    std::swap(intersection.target, intersection.target_other);
    intersections.push_back(intersection);
  };
  if (intersection.target_other.label == tf::topo_type::face) {
    make_push(intersection);
  } else if (intersection.target_other.label == tf::topo_type::vertex) {
    // all polygons containing the vertex will be processed
    Index pt_id =
        faces[intersection.object_other][intersection.target_other.id];
    for (auto poly_id : fe[pt_id]) {
      Index n_pt_id =
          std::find(faces[poly_id].begin(), faces[poly_id].end(), pt_id) -
          faces[poly_id].begin();
      auto n_intersection = intersection;
      n_intersection.object_other = poly_id;
      n_intersection.target_other.id = n_pt_id;
      make_push(n_intersection);
    }
  } else if (intersection.target_other.label == tf::topo_type::edge) {
    // we only process the neighbors further down
    make_push(intersection);
    auto N = faces[intersection.object_other].size();
    Index e0 = faces[intersection.object_other][intersection.target_other.id];
    Index e1 = faces[intersection.object_other][tf::circular_increment<Index>(
        intersection.target_other.id, Index(N))];
    if (mel[intersection.object_other][intersection.target_other.id]
            .is_simple()) {
      Index n_poly_id =
          mel[intersection.object_other][intersection.target_other.id]
              .face_peer;
      Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
      auto n_intersection = intersection;
      n_intersection.object_other = n_poly_id;
      n_intersection.target_other.id = n_e;
      make_push(n_intersection);
    } else if (!mel[intersection.object_other][intersection.target_other.id]
                    .is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe, faces, intersection.object_other, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_poly_id : neighbors) {
        Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
        auto n_intersection = intersection;
        n_intersection.object_other = n_poly_id;
        n_intersection.target_other.id = n_e;
        make_push(n_intersection);
      }
    }
  }
}

template <typename Policy, typename Index, typename Policy1, typename Policy2>
auto duplicate_self_intersection(
    const tf::faces<Policy> &faces,
    tf::intersect::intersection<Index> intersection,
    const tf::face_membership_like<Policy1> &fe,
    const tf::manifold_edge_link_like<Policy2> &mel,
    tf::buffer<tf::intersect::intersection<Index>> &intersections) {

  if (intersection.target.label == tf::topo_type::face) {
    duplicate_self_intersection2(faces, intersection, fe, mel, intersections);

  } else if (intersection.target.label == tf::topo_type::vertex) {
    // all polygons containing the vertex will be processed
    Index pt_id = faces[intersection.object][intersection.target.id];
    for (auto poly_id : fe[pt_id]) {
      Index n_pt_id =
          std::find(faces[poly_id].begin(), faces[poly_id].end(), pt_id) -
          faces[poly_id].begin();
      auto n_intersection = intersection;
      n_intersection.object = poly_id;
      n_intersection.target.id = n_pt_id;
      duplicate_self_intersection2(faces, n_intersection, fe, mel,
                                   intersections);
    }
  } else if (intersection.target.label == tf::topo_type::edge) {
    // we only process the neighbors further down
    duplicate_self_intersection2(faces, intersection, fe, mel, intersections);
    auto N = faces[intersection.object].size();
    Index e0 = faces[intersection.object][intersection.target.id];
    Index e1 =
        faces[intersection.object]
             [tf::circular_increment<Index>(intersection.target.id, Index(N))];
    if (mel[intersection.object][intersection.target.id].is_simple()) {
      Index n_poly_id =
          mel[intersection.object][intersection.target.id].face_peer;
      Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
      auto n_intersection = intersection;
      n_intersection.object = n_poly_id;
      n_intersection.target.id = n_e;
      duplicate_self_intersection2(faces, n_intersection, fe, mel,
                                   intersections);
    } else if (!mel[intersection.object][intersection.target.id]
                    .is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe, faces, intersection.object, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_poly_id : neighbors) {
        Index n_e = tf::edge_id_in_face(e1, e0, faces[n_poly_id]);
        auto n_intersection = intersection;
        n_intersection.object = n_poly_id;
        n_intersection.target.id = n_e;
        duplicate_self_intersection2(faces, n_intersection, fe, mel,
                                     intersections);
      }
    }
  }
}
} // namespace tf::intersect
