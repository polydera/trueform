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

#include "../core/aabb_from.hpp"
#include "../core/algorithm/circular_decrement.hpp"
#include "../core/algorithm/circular_increment.hpp"
#include "../core/buffer.hpp"
#include "../core/classify.hpp"
#include "../core/empty_aabb.hpp"
#include "../core/faces.hpp"
#include "../core/intersects.hpp"
#include "../core/points.hpp"
#include <algorithm>

namespace tf {

/// @ingroup topology_planar
/// @brief Patches holes into a face to create a single contour.
///
/// Given an outer loop (face boundary) and inner loops (holes), creates
/// a single polygon by connecting holes to the outer boundary. Uses a
/// greedy algorithm that connects each hole's leftmost vertex to the
/// nearest visible vertex on the face.
///
/// @tparam Index The integer type for indices.
template <typename Index> class hole_patcher {
public:
  /// @brief Build a patched face from outer loop and holes.
  /// @tparam Range0 The loop range type.
  /// @tparam Policy0 The holes policy type.
  /// @tparam Policy1 The points policy type.
  /// @param loop The outer loop (face boundary).
  /// @param holes The inner loops (holes).
  /// @param points The vertex positions.
  template <typename Range0, typename Policy0, typename Policy1>
  auto build(const Range0 &loop, const tf::faces<Policy0> &holes,
             const tf::points<Policy1> &points) {
    clear();
    /*std::cout << "hole patcher:" << std::endl;*/
    /*for(auto e:loop)*/
    /*  std::cout << e << ", ";*/
    /*std::cout << std::endl;*/
    /*for(auto loop:holes) {*/
    /*  std::cout << "  ";*/
    /*for(auto e:loop)*/
    /*  std::cout << e << ", ";*/
    /*std::cout << std::endl;*/
    /*}*/
    patch_holes(loop, holes, points);
    fill_buffer();
  }

  /// @brief Get the resulting patched face.
  /// @return A range of vertex indices forming the patched face.
  auto face() const { return tf::make_range(_work_buffer); }

  /// @brief Clear all internal state.
  auto clear() {
    _nodes.clear();
    _face_heap.clear();
    _work_buffer.clear();
  }

private:
  class node_t {
  public:
    Index id;
    Index pt_id;
    Index next;
    Index prev;
  };
  auto next(const node_t &node) const -> const node_t & {
    return _nodes[node.next];
  }

  auto prev(const node_t &node) const -> const node_t & {
    return _nodes[node.prev];
  }

  auto next(const node_t &node) -> node_t & { return _nodes[node.next]; }

  auto prev(const node_t &node) -> node_t & { return _nodes[node.prev]; }

  template <typename Policy>
  auto point(const tf::points<Policy> &points, const node_t &node) const {
    return points[node.pt_id].template as<double>();
  }

  auto fill_buffer() {
    auto current = _nodes.front();
    _work_buffer.clear();
    do {
      _work_buffer.push_back(current.pt_id);
      current = next(current);
    } while (current.id != _nodes.front().id);
  };

  template <typename Range0, typename Policy>
  auto push_loop(const Range0 &loop, const tf::points<Policy> &points,
                 Index axis) {
    Index size = loop.size();
    Index initial = _nodes.size();
    std::pair<tf::coordinate_type<Policy>, Index> min{points[loop[axis]][0],
                                                      Index(initial)};
    for (Index i = 0; i < size; ++i) {
      _nodes.push_back(node_t{initial + i, loop[i],
                              tf::circular_increment(i, size) + initial,
                              tf::circular_decrement(i, size) + initial});
      min = std::min(min,
                     std::make_pair(points[loop[i]][axis], Index(i + initial)));
    };
    return min.second;
  }

  template <typename Policy>
  auto fill_heap(const node_t loop, const node_t hole,
                 const tf::points<Policy> &points) {
    _face_heap.clear();
    node_t current = loop;
    do {
      if (point(points, current)[0] <= point(points, hole)[0])
        _face_heap.push_back(
            {(point(points, current) - point(points, hole)).length2(),
             current.id});
      current = next(current);
    } while (current.id != loop.id);
  }

  template <typename Policy>
  auto test_attachment(const node_t face_node, const node_t hole_min_node,

                       const tf::points<Policy> &points) {
    if (face_node.pt_id == hole_min_node.pt_id)
      return true;
    if (point(points, face_node) == point(points, hole_min_node))
      return true;
    if (tf::classify(point(points, hole_min_node),
                     tf::make_wedge(point(points, face_node),
                                    point(points, next(face_node)),
                                    point(points, prev(face_node)))) !=
        tf::strict_containment::inside)
      return false;

    auto seg0 = tf::make_segment_between_points(point(points, hole_min_node),
                                                point(points, face_node));
    node_t prev_n = prev(face_node);
    auto side_prev = tf::classify(point(points, prev_n), seg0);
    node_t current_n = face_node;
    do {
      auto side_n = tf::classify(point(points, current_n), seg0);
      if (side_n != side_prev && prev_n.pt_id != face_node.pt_id &&
          current_n.pt_id != face_node.pt_id) {
        auto seg1 = tf::make_segment_between_points(point(points, prev_n),
                                                    point(points, current_n));
        if (tf::intersects(seg0, seg1))
          return false;
      }
      prev_n = current_n;
      current_n = next(current_n);
      side_prev = side_n;
    } while (current_n.id != face_node.id);
    return true;
  }

  auto patch_hole(const node_t face_node_, const node_t hole_node_) {
    auto &hole_node = _nodes[hole_node_.id];
    auto &face_node = _nodes[face_node_.id];
    // (a, b, c), (a, d, e) -> (a, b, c, a, d, e)
    if (face_node.pt_id == hole_node.pt_id) {
      prev(face_node).next = hole_node.id;
      auto face_node_prev = face_node.prev;
      face_node.prev = hole_node.prev;
      prev(hole_node).next = face_node.id;
      hole_node.prev = face_node_prev;
      return;
    }
    // a -- e
    // (a, b, c), (e, f, g) -> (a, b, c, a, e, f, g, e)
    node_t new_e = hole_node;
    new_e.id = _nodes.size();
    new_e.next = face_node.id;
    new_e.prev = hole_node.prev;
    prev(hole_node).next = new_e.id;
    //
    node_t new_a = face_node;
    new_a.id = new_e.id + 1;
    new_a.next = hole_node.id;
    new_a.prev = face_node.prev;
    prev(face_node).next = new_a.id;
    face_node.prev = new_e.id;
    hole_node.prev = new_a.id;
    //
    _nodes.push_back(new_e);
    _nodes.push_back(new_a);
  }

  template <typename Policy>
  auto patch_hole(const node_t loop, const node_t min_hole_node,
                  const tf::points<Policy> &points, Index axis) {
    _face_heap.clear();
    fill_heap(loop, min_hole_node, points);
    auto compare = [&](const auto &x0, const auto &x1) {
      return std::make_pair(x0.d2, -point(points, _nodes[x0.id])[axis]) >
             std::make_pair(x1.d2, -point(points, _nodes[x1.id])[axis]);
    };

    std::make_heap(_face_heap.begin(), _face_heap.end(), compare);
    while (_face_heap.size()) {
      std::pop_heap(_face_heap.begin(), _face_heap.end(), compare);
      auto current_node = _nodes[_face_heap.back().id];
      _face_heap.pop_back();
      if (test_attachment(current_node, min_hole_node, points)) {
        patch_hole(current_node, min_hole_node);
        return;
      }
    }
  }

  template <typename Policy0, typename Policy1>
  auto compute_axis(const tf::faces<Policy0> &holes,
                    const tf::points<Policy1> &points) {
    auto aabb = tf::make_empty_aabb<double, 2>();
    for (const auto &hole : holes)
      tf::aabb_union_inplace(aabb,
                             tf::aabb_from(tf::make_polygon(hole, points)));
    auto diag = aabb.diagonal();
    Index axis = std::max_element(diag.begin(), diag.end()) - diag.begin();
    return axis;
  }

  template <typename Range0, typename Policy0, typename Policy1>
  auto patch_holes(const Range0 &loop, const tf::faces<Policy0> &holes,
                   const tf::points<Policy1> &points) {
    Index axis = compute_axis(holes, points);
    push_loop(loop, points, axis);
    _work_buffer.reserve(holes.size());
    for (const auto &hole : holes)
      _work_buffer.push_back(push_loop(hole, points, axis));

    std::sort(_work_buffer.begin(), _work_buffer.end(),
              [&](Index id0, Index id1) {
                return points[_nodes[id0].pt_id][axis] <
                       points[_nodes[id1].pt_id][axis];
              });

    for (Index hole_node_id : _work_buffer)
      patch_hole(_nodes[0], _nodes[hole_node_id], points, axis);
  }

  struct heap_node_t {
    double d2;
    Index id;
  };

  tf::buffer<node_t> _nodes;
  tf::buffer<Index> _work_buffer;
  tf::buffer<heap_node_t> _face_heap;
};
} // namespace tf
