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
#include "./delaunay_execution_tuning.hpp"
#include "./delaunay_frontier.hpp"
#include "./partition_morton_sites.hpp"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace tf::topology::cdt {

/// Owns the transient navigation vocabulary of the Guibas–Stolfi operation.
/// Points, predicates, and edge storage remain policies supplied by the
/// caller; the operation carries no result cache and has no lifetime beyond one
/// build phase.
template <typename Index, typename Points, typename Edges, typename KeyBuffer,
          typename Orient, typename Incircle, bool SortTerminalSites = false>
class delaunay_divide_and_conquer {
  enum class merge_side { lower, upper };

  struct merge_candidate {
    Index edge;
    bool visible;
  };

public:
  delaunay_divide_and_conquer(Points &points, Edges &edges,
                              const KeyBuffer &keys, Index none, Orient orient,
                              Incircle incircle)
      : _points(points), _edges(edges), _keys(keys), _none(none),
        _orient(std::move(orient)), _incircle(std::move(incircle)) {}

  auto build() -> delaunay_build_result<Index> {
    return {twin(build_root(0, _points.size()))};
  }

  auto build_partition(std::size_t first, std::size_t last)
      -> delaunay_frontier<Index> {
    return divide_morton_partition(first, last);
  }

  auto join_partitions(delaunay_frontier<Index> lower,
                       delaunay_frontier<Index> upper, delaunay_axis axis)
      -> delaunay_frontier<Index> {
    const auto joined = merge_boundaries(lower[axis], upper[axis]);
    if (axis == delaunay_axis::x)
      return complete_frontier<delaunay_axis::x>(twin(joined.minimum), joined);
    return complete_frontier<delaunay_axis::y>(twin(joined.minimum), joined);
  }

private:
  static auto twin(Index edge) -> Index { return edge ^ Index(1); }

  auto origin(Index edge) const -> Index {
    return _edges[std::size_t(edge)].vertex;
  }

  auto target(Index edge) const -> Index { return origin(twin(edge)); }

  auto origin_successor(Index edge) const -> Index {
    return _edges[std::size_t(edge)].prev;
  }

  auto origin_predecessor(Index edge) const -> Index {
    return _edges[std::size_t(edge)].next;
  }

  auto left_face_successor(Index edge) const -> Index {
    return origin_predecessor(twin(edge));
  }

  auto point(Index site) const { return _points[std::size_t(site)]; }

  auto append_edge(Index from, Index to) -> Index {
    const Index edge = static_cast<Index>(_edges.append_pair());
    decltype(auto) forward = _edges[std::size_t(edge)];
    forward.vertex = from;
    forward.prev = edge;
    forward.next = edge;

    const Index reverse_edge = twin(edge);
    decltype(auto) reverse = _edges[std::size_t(reverse_edge)];
    reverse.vertex = to;
    reverse.prev = reverse_edge;
    reverse.next = reverse_edge;
    return edge;
  }

  auto link_origin_successor(Index edge, Index successor) -> void {
    _edges[std::size_t(edge)].prev = successor;
    _edges[std::size_t(successor)].next = edge;
  }

  auto exchange_origin_successors(Index first, Index second) -> void {
    const Index after_first = origin_successor(first);
    const Index after_second = origin_successor(second);
    link_origin_successor(first, after_second);
    link_origin_successor(second, after_first);
  }

  auto bridge_left_faces(Index lower, Index upper) -> Index {
    const Index bridge = append_edge(target(lower), origin(upper));
    exchange_origin_successors(bridge, left_face_successor(lower));
    exchange_origin_successors(twin(bridge), upper);
    return bridge;
  }

  auto discard_edge(Index edge) -> void {
    const Index reverse = twin(edge);
    exchange_origin_successors(edge, origin_predecessor(edge));
    exchange_origin_successors(reverse, origin_predecessor(reverse));
    for (const Index dart : {edge, reverse})
      _edges[std::size_t(dart)].vertex = _none;
  }

  auto site_is_left_of(Index site, Index edge) const -> bool {
    return _orient(origin(edge), target(edge), site) > 0;
  }

  auto site_is_right_of(Index site, Index edge) const -> bool {
    return _orient(origin(edge), target(edge), site) < 0;
  }

  auto seed_two(std::size_t first) -> delaunay_boundary<Index> {
    const Index edge =
        append_edge(static_cast<Index>(first), static_cast<Index>(first + 1));
    return {edge, twin(edge)};
  }

  auto seed_three(std::size_t first) -> delaunay_boundary<Index> {
    const Index a = static_cast<Index>(first);
    const Index b = static_cast<Index>(first + 1);
    const Index c = static_cast<Index>(first + 2);
    const Index ab = append_edge(a, b);
    const Index bc = append_edge(b, c);
    exchange_origin_successors(twin(ab), bc);

    const int orientation = _orient(a, b, c);
    if (orientation > 0) {
      bridge_left_faces(bc, ab);
      return {ab, twin(bc)};
    }
    if (orientation < 0) {
      const Index ca = bridge_left_faces(bc, ab);
      return {twin(ca), ca};
    }
    return {ab, twin(bc)};
  }

  auto triangulate_sorted_range(std::size_t first, std::size_t last)
      -> delaunay_boundary<Index> {
    switch (last - first) {
    case 2:
      return seed_two(first);
    case 3:
      return seed_three(first);
    default:
      break;
    }

    const std::size_t cut = first + (last - first) / 2;
    auto lower = triangulate_sorted_range(first, cut);
    auto upper = triangulate_sorted_range(cut, last);
    return merge_boundaries(lower, upper);
  }

  auto find_common_tangent(delaunay_boundary<Index> lower,
                           delaunay_boundary<Index> upper)
      -> std::pair<Index, Index> {
    Index lower_edge = lower.maximum;
    Index upper_edge = upper.minimum;
    for (;;) {
      if (site_is_left_of(origin(upper_edge), lower_edge)) {
        lower_edge = left_face_successor(lower_edge);
        continue;
      }
      if (site_is_right_of(origin(lower_edge), upper_edge)) {
        upper_edge = origin_successor(twin(upper_edge));
        continue;
      }
      return {lower_edge, upper_edge};
    }
  }

  template <merge_side Side> auto first_candidate(Index base) const -> Index {
    if constexpr (Side == merge_side::lower)
      return origin_successor(twin(base));
    return origin_predecessor(base);
  }

  template <merge_side Side>
  auto following_candidate(Index edge) const -> Index {
    if constexpr (Side == merge_side::lower)
      return origin_successor(edge);
    return origin_predecessor(edge);
  }

  template <merge_side Side>
  auto prune_candidate(Index base) -> merge_candidate {
    merge_candidate candidate{first_candidate<Side>(base), false};
    candidate.visible = site_is_right_of(target(candidate.edge), base);
    if (!candidate.visible)
      return candidate;

    bool removed = false;
    for (;;) {
      const Index beyond = following_candidate<Side>(candidate.edge);
      if (!_incircle(target(base), origin(base), target(candidate.edge),
                     target(beyond)))
        break;
      discard_edge(candidate.edge);
      candidate.edge = beyond;
      removed = true;
    }
    if (removed)
      candidate.visible = site_is_right_of(target(candidate.edge), base);
    return candidate;
  }

  auto merge_boundaries(delaunay_boundary<Index> lower,
                        delaunay_boundary<Index> upper)
      -> delaunay_boundary<Index> {
    const auto tangent = find_common_tangent(lower, upper);
    const Index lower_inner = tangent.first;
    const Index upper_inner = tangent.second;
    Index base = bridge_left_faces(twin(upper_inner), lower_inner);

    if (origin(lower_inner) == origin(lower.minimum))
      lower.minimum = twin(base);
    if (origin(upper_inner) == origin(upper.maximum))
      upper.maximum = base;

    for (;;) {
      const auto lower_candidate = prune_candidate<merge_side::lower>(base);
      const auto upper_candidate = prune_candidate<merge_side::upper>(base);
      if (!lower_candidate.visible && !upper_candidate.visible)
        return {lower.minimum, upper.maximum};

      const bool advance_upper =
          !lower_candidate.visible ||
          (upper_candidate.visible &&
           _incircle(target(lower_candidate.edge), origin(lower_candidate.edge),
                     origin(upper_candidate.edge),
                     target(upper_candidate.edge)));
      base = advance_upper
                 ? bridge_left_faces(upper_candidate.edge, twin(base))
                 : bridge_left_faces(twin(base), twin(lower_candidate.edge));
    }
  }

  template <delaunay_axis Axis, typename Point>
  static auto precedes_minimum(const Point &candidate, const Point &current)
      -> bool {
    if constexpr (Axis == delaunay_axis::x)
      return candidate[0] < current[0] ||
             (candidate[0] == current[0] && candidate[1] < current[1]);
    return candidate[1] < current[1] ||
           (candidate[1] == current[1] && candidate[0] > current[0]);
  }

  template <delaunay_axis Axis, typename Point>
  static auto follows_maximum(const Point &candidate, const Point &current)
      -> bool {
    if constexpr (Axis == delaunay_axis::x)
      return candidate[0] > current[0] ||
             (candidate[0] == current[0] && candidate[1] < current[1]);
    return candidate[1] > current[1] ||
           (candidate[1] == current[1] && candidate[0] > current[0]);
  }

  template <delaunay_axis Axis>
  auto consider_frontier_edge(delaunay_boundary<Index> &boundary,
                              Index outer) const -> void {
    if (precedes_minimum<Axis>(point(target(outer)),
                               point(origin(boundary.minimum))))
      boundary.minimum = twin(outer);
    if (follows_maximum<Axis>(point(origin(outer)),
                              point(origin(boundary.maximum))))
      boundary.maximum = outer;
  }

  auto measure_frontier(Index outer_seed) const -> delaunay_frontier<Index> {
    delaunay_frontier<Index> result;
    result[delaunay_axis::x] = {twin(outer_seed), outer_seed};
    result[delaunay_axis::y] = {twin(outer_seed), outer_seed};
    for (Index outer = left_face_successor(outer_seed); outer != outer_seed;
         outer = left_face_successor(outer)) {
      consider_frontier_edge<delaunay_axis::x>(result[delaunay_axis::x], outer);
      consider_frontier_edge<delaunay_axis::y>(result[delaunay_axis::y], outer);
    }
    return result;
  }

  template <delaunay_axis JoinedAxis>
  auto complete_frontier(Index outer_seed,
                         delaunay_boundary<Index> joined) const
      -> delaunay_frontier<Index> {
    constexpr delaunay_axis scanned_axis = perpendicular(JoinedAxis);
    delaunay_frontier<Index> result;
    result[JoinedAxis] = joined;
    result[scanned_axis] = {twin(outer_seed), outer_seed};
    for (Index outer = left_face_successor(outer_seed); outer != outer_seed;
         outer = left_face_successor(outer))
      consider_frontier_edge<scanned_axis>(result[scanned_axis], outer);
    return result;
  }

  auto order_and_triangulate(std::size_t first, std::size_t last)
      -> delaunay_boundary<Index> {
    if constexpr (SortTerminalSites) {
      std::sort(_points.begin() + std::ptrdiff_t(first),
                _points.begin() + std::ptrdiff_t(last),
                [](const auto &a, const auto &b) {
                  return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
                });
    }
    return triangulate_sorted_range(first, last);
  }

  auto build_terminal(std::size_t first, std::size_t last)
      -> delaunay_frontier<Index> {
    return measure_frontier(twin(order_and_triangulate(first, last).minimum));
  }

  /// The root has no sibling, so its frontier is measured for no one: only an
  /// outer dart leaves the operation, and the boundary a merge already ends
  /// holding is one.
  auto build_root(std::size_t first, std::size_t last) -> Index {
    if (last - first <= delaunay_execution_tuning::leaf_sites)
      return order_and_triangulate(first, last).minimum;

    const auto partition = partition_morton_sites(_keys, first, last);
    if (!partition)
      return order_and_triangulate(first, last).minimum;

    auto lower = divide_morton_partition(first, partition->cut);
    auto upper = divide_morton_partition(partition->cut, last);
    return merge_boundaries(lower[partition->axis], upper[partition->axis])
        .minimum;
  }

  auto divide_morton_partition(std::size_t first, std::size_t last)
      -> delaunay_frontier<Index> {
    if (last - first <= delaunay_execution_tuning::leaf_sites)
      return build_terminal(first, last);

    const auto partition = partition_morton_sites(_keys, first, last);
    if (!partition)
      return build_terminal(first, last);

    auto lower = divide_morton_partition(first, partition->cut);
    auto upper = divide_morton_partition(partition->cut, last);
    return join_partitions(lower, upper, partition->axis);
  }

  Points &_points;
  Edges &_edges;
  const KeyBuffer &_keys;
  Index _none;
  Orient _orient;
  Incircle _incircle;
};

template <typename Index, bool SortTerminalSites = false, typename Points,
          typename Edges, typename KeyBuffer, typename Orient,
          typename Incircle>
auto build_divide_and_conquer(Points &points, Edges &edges,
                              const KeyBuffer &keys, Index none, Orient orient,
                              Incircle incircle)
    -> delaunay_build_result<Index> {
  return delaunay_divide_and_conquer<Index, Points, Edges, KeyBuffer, Orient,
                                     Incircle, SortTerminalSites>(
             points, edges, keys, none, std::move(orient), std::move(incircle))
      .build();
}

} // namespace tf::topology::cdt
