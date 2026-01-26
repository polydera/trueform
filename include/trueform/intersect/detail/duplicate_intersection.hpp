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
#include "../../topology/edge_membership_like.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../../topology/manifold_edge_link_like.hpp"
#include "../types/intersection.hpp"
#include "../types/tagged_intersection.hpp"

namespace tf::intersect {
template <typename Policy, typename Index, typename Policy1, typename Policy2>
auto duplicate_intersection(
    const tf::faces<Policy> &faces,
    tf::intersect::tagged_intersection<Index> intersection,
    const tf::face_membership_like<Policy1> &fe,
    const tf::manifold_edge_link_like<Policy2> &mel,
    tf::buffer<tf::intersect::tagged_intersection<Index>> &intersections) {
  auto make_push = [&](tf::intersect::tagged_intersection<Index> intersection) {
    intersections.push_back(intersection);
    intersection.tag = 1;
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

template <typename Policy0, typename Policy1, typename Index, typename Policy2,
          typename Policy3, typename Policy4, typename Policy5>
auto duplicate_intersection(
    const tf::faces<Policy0> &faces0, const tf::faces<Policy1> &faces1,
    tf::intersect::tagged_intersection<Index> intersection,
    const tf::face_membership_like<Policy2> &fe0,
    const tf::manifold_edge_link_like<Policy3> &mel0,
    const tf::face_membership_like<Policy4> &fe1,
    const tf::manifold_edge_link_like<Policy5> &mel1,
    tf::buffer<tf::intersect::tagged_intersection<Index>> &intersections) {

  if (intersection.target.label == tf::topo_type::face) {
    duplicate_intersection(faces1, intersection, fe1, mel1, intersections);

  } else if (intersection.target.label == tf::topo_type::vertex) {
    // all polygons containing the vertex will be processed
    Index pt_id = faces0[intersection.object][intersection.target.id];
    for (auto poly_id : fe0[pt_id]) {
      Index n_pt_id =
          std::find(faces0[poly_id].begin(), faces0[poly_id].end(), pt_id) -
          faces0[poly_id].begin();
      auto n_intersection = intersection;
      n_intersection.object = poly_id;
      n_intersection.target.id = n_pt_id;
      duplicate_intersection(faces1, n_intersection, fe1, mel1, intersections);
    }
  } else if (intersection.target.label == tf::topo_type::edge) {
    // we only process the neighbors further down
    duplicate_intersection(faces1, intersection, fe1, mel1, intersections);
    auto N0 = faces0[intersection.object].size();
    Index e0 = faces0[intersection.object][intersection.target.id];
    Index e1 = faces0[intersection.object][tf::circular_increment<Index>(
        intersection.target.id, Index(N0))];
    if (mel0[intersection.object][intersection.target.id].is_simple()) {
      Index n_poly_id =
          mel0[intersection.object][intersection.target.id].face_peer;
      Index n_e = tf::edge_id_in_face(e1, e0, faces0[n_poly_id]);
      auto n_intersection = intersection;
      n_intersection.object = n_poly_id;
      n_intersection.target.id = n_e;
      duplicate_intersection(faces1, n_intersection, fe1, mel1, intersections);
    } else if (!mel0[intersection.object][intersection.target.id]
                    .is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe0, faces0, intersection.object, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_poly_id : neighbors) {
        Index n_e = tf::edge_id_in_face(e1, e0, faces0[n_poly_id]);
        auto n_intersection = intersection;
        n_intersection.object = n_poly_id;
        n_intersection.target.id = n_e;
        duplicate_intersection(faces1, n_intersection, fe1, mel1,
                               intersections);
      }
    }
  }
}

namespace detail {
template <typename Index, typename Policy>
auto duplicate_intersection1(
    intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em,
    tf::buffer<intersect::intersection<Index>> &buffer) {
  auto push_f = [&](auto i) {
    buffer.push_back(i);
    std::swap(i.target, i.target_other);
    std::swap(i.object, i.object_other);
    buffer.push_back(i);
  };
  if (i.target_other.label == tf::topo_type::edge) {
    push_f(i);
  } else if (i.target_other.label == tf::topo_type::vertex) {
    for (auto edge_id1 : em[i.target_other.id]) {
      i.object_other = edge_id1;
      push_f(i);
    }
  }
}
} // namespace detail
template <typename Index, typename Policy>
auto duplicate_intersection(
    intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em,
    tf::buffer<intersect::intersection<Index>> &buffer) {
  if (i.target.label == tf::topo_type::edge) {
    detail::duplicate_intersection1(i, em, buffer);
  } else if (i.target.label == tf::topo_type::vertex) {
    for (auto edge_id0 : em[i.target.id]) {
      i.object = edge_id0;
      detail::duplicate_intersection1(i, em, buffer);
    }
  }
}

} // namespace tf::intersect
