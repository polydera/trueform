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

#include "../core/algorithm/circular_decrement.hpp"
#include "../core/algorithm/circular_increment.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/faces.hpp"
#include "../core/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/meta.hpp"
#include "../exact/orient2d.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/segments_cross.hpp"
#include <algorithm>
#include <limits>

namespace tf {

/// @ingroup topology_planar
/// @brief Patches holes into a face to create a single contour.
///
/// Given an outer loop (face boundary) and inner loops (holes), creates
/// a single polygon by connecting holes to the outer boundary. Every
/// geometric predicate is exact on the `Int` lattice; floating-point points
/// are converted onto it, integral ones consumed as they are.
///
/// If a hole shares a vertex with the face, splices directly there
/// (no bridge edges). Otherwise, uses a heap-based algorithm to find
/// the nearest visible vertex.
///
/// @tparam Int The lattice the exact predicates run on. `build` accepts any
///   coordinate type, so the caller states it.
template <typename Index, typename Int> class hole_patcher {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

public:
  template <typename Range, typename Policy0, typename Policy1>
  auto build(const Range &loop, const tf::faces<Policy0> &holes,
             const tf::points<Policy1> &points) {
    clear();
    if (!holes.size())
      return;
    using coord_t = tf::coordinate_type<Policy1>;
    if constexpr (std::is_integral_v<coord_t>) {
      build_impl(loop, holes, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      build_impl(loop, holes, int_pts);
    }
  }

  auto face() const { return tf::make_range(_work_buffer); }

  auto clear() {
    _nodes.clear();
    _face_heap.clear();
    _work_buffer.clear();
  }

private:
  struct node_t {
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
  auto pt(const tf::points<Policy> &points, const node_t &node) const {
    auto &&p = points[node.pt_id];
    return tf::exact::pt2<Int>{p[0], p[1]};
  }

  auto fill_buffer() {
    auto current = _nodes.front();
    _work_buffer.clear();
    do {
      _work_buffer.push_back(current.pt_id);
      current = next(current);
    } while (current.id != _nodes.front().id);
  }

  template <typename Range, typename Policy>
  auto push_loop(const Range &loop, const tf::points<Policy> &points,
                 int axis) -> Index {
    Index size = loop.size();
    Index initial = _nodes.size();
    auto min_val = points[loop[0]][axis];
    Index min_id = initial;
    for (Index i = 0; i < size; ++i) {
      _nodes.push_back(
          {initial + i, loop[i], tf::circular_increment(i, size) + initial,
           tf::circular_decrement(i, size) + initial});
      auto v = points[loop[i]][axis];
      if (v < min_val) {
        min_val = v;
        min_id = initial + i;
      }
    }
    return min_id;
  }

  // Try to find a shared pt_id between hole and face. Returns the
  // face node index and hole node index if found, or (-1, -1).
  auto find_shared_vertex(Index face_start, Index hole_start)
      -> std::pair<Index, Index> {
    auto h = _nodes[hole_start];
    do {
      auto f = _nodes[face_start];
      do {
        if (f.pt_id == h.pt_id)
          return {f.id, h.id};
        f = next(f);
      } while (f.id != face_start);
      h = next(h);
    } while (h.id != hole_start);
    return {Index(-1), Index(-1)};
  }

  template <typename Policy>
  auto fill_heap(const node_t &loop, const node_t &hole,
                 const tf::points<Policy> &points, int axis) {
    _face_heap.clear();
    auto hp = pt(points, hole);
    node_t current = loop;
    do {
      auto cp = pt(points, current);
      if (cp[axis] <= hp[axis])
        _face_heap.push_back({tf::exact::distance2(cp, hp), current.id});
      current = next(current);
    } while (current.id != loop.id);
  }

  template <typename Policy>
  auto test_attachment(const node_t &face_node, const node_t &hole_min_node,
                       const tf::points<Policy> &points) -> bool {
    if (face_node.pt_id == hole_min_node.pt_id)
      return true;

    auto fp = pt(points, face_node);
    auto hp = pt(points, hole_min_node);
    if (fp[0] == hp[0] && fp[1] == hp[1])
      return true;

    // Wedge test: is hole point inside wedge at face_node?
    auto np = pt(points, next(face_node));
    auto pp = pt(points, prev(face_node));
    auto o_next = tf::exact::orient2d(fp, np, hp);
    auto o_prev = tf::exact::orient2d(fp, pp, hp);
    auto o_wedge = tf::exact::orient2d(fp, np, pp);

    bool in_wedge;
    if (o_wedge >= 0) // convex wedge
      in_wedge = (o_next > 0 && o_prev < 0);
    else // reflex wedge
      in_wedge = (o_next > 0 || o_prev < 0);

    if (!in_wedge)
      return false;

    // Visibility test: does bridge (hole→face) cross any boundary edge?
    node_t prev_n = prev(face_node);
    node_t current_n = face_node;
    do {
      if (current_n.pt_id != face_node.pt_id &&
          prev_n.pt_id != face_node.pt_id) {
        auto a = pt(points, prev_n);
        auto b = pt(points, current_n);
        if (tf::exact::segments_cross(hp, fp, a, b))
          return false;
      }
      prev_n = current_n;
      current_n = next(current_n);
    } while (current_n.id != face_node.id);

    return true;
  }

  auto patch_hole(const node_t &face_node_, const node_t &hole_node_) {
    auto &hole_node = _nodes[hole_node_.id];
    auto &face_node = _nodes[face_node_.id];

    if (face_node.pt_id == hole_node.pt_id) {
      prev(face_node).next = hole_node.id;
      auto face_node_prev = face_node.prev;
      face_node.prev = hole_node.prev;
      prev(hole_node).next = face_node.id;
      hole_node.prev = face_node_prev;
      return;
    }

    node_t new_e = hole_node;
    new_e.id = _nodes.size();
    new_e.next = face_node.id;
    new_e.prev = hole_node.prev;
    prev(hole_node).next = new_e.id;

    node_t new_a = face_node;
    new_a.id = new_e.id + 1;
    new_a.next = hole_node.id;
    new_a.prev = face_node.prev;
    prev(face_node).next = new_a.id;
    face_node.prev = new_e.id;
    hole_node.prev = new_a.id;

    _nodes.push_back(new_e);
    _nodes.push_back(new_a);
  }

  template <typename Policy>
  auto patch_hole(const node_t &loop, const node_t &min_hole_node,
                  const tf::points<Policy> &points, int axis) {
    // Try shared vertex first
    auto [f_id, h_id] = find_shared_vertex(loop.id, min_hole_node.id);
    if (f_id != Index(-1)) {
      patch_hole(_nodes[f_id], _nodes[h_id]);
      return;
    }

    // Heap-based fallback
    fill_heap(loop, min_hole_node, points, axis);

    auto compare = [&](const auto &x0, const auto &x1) {
      auto p0 = pt(points, _nodes[x0.id]);
      auto p1 = pt(points, _nodes[x1.id]);
      return std::make_pair(x0.d2, -T1(p0[axis])) >
             std::make_pair(x1.d2, -T1(p1[axis]));
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
                    const tf::points<Policy1> &points) -> int {
    Int min0 = std::numeric_limits<Int>::max();
    Int max0 = std::numeric_limits<Int>::min();
    Int min1 = std::numeric_limits<Int>::max();
    Int max1 = std::numeric_limits<Int>::min();
    for (const auto &hole : holes) {
      for (auto id : hole) {
        auto &&p = points[id];
        min0 = std::min(min0, Int(p[0]));
        max0 = std::max(max0, Int(p[0]));
        min1 = std::min(min1, Int(p[1]));
        max1 = std::max(max1, Int(p[1]));
      }
    }
    return (max0 - min0 >= max1 - min1) ? 0 : 1;
  }

  template <typename Range, typename Policy0, typename Policy1>
  auto patch_holes(const Range &loop, const tf::faces<Policy0> &holes,
                   const tf::points<Policy1> &points) {
    int axis = compute_axis(holes, points);
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

  template <typename Range, typename Policy0, typename Policy1>
  auto build_impl(const Range &loop, const tf::faces<Policy0> &holes,
                  const tf::points<Policy1> &points) {
    patch_holes(loop, holes, points);
    fill_buffer();
  }

  struct heap_node_t {
    T2 d2;
    Index id;
  };

  tf::buffer<node_t> _nodes;
  tf::buffer<Index> _work_buffer;
  tf::buffer<heap_node_t> _face_heap;
};
} // namespace tf
