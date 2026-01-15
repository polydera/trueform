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
#include "../../core/classify.hpp"
#include "../../core/point_view.hpp"
#include <cstdint>
#include <numeric>

namespace tf::geom {
/**
 * @brief Performs earcutting on a polygon
 *
 * @tparam Index Type to use for indices
 * @tparam ForceAllPoints Should the algoriothm force inclusion of
 *                        all points on failure. This is needed when
 *                        triangulating intersected loops that will later
 *                        be merged along the seem (both sides need the
 *                        same, all points)
 */
template <typename Index, bool ForceAllPoints = true> class earcutter {

public:
  template <typename Range> auto operator()(const Range &points) {
    clear();
    auto n_pts = points.size();
    if (n_pts <= 3) {
      if (n_pts <= 1)
        return false;
      _triangles.allocate(3);
      std::iota(_triangles.begin(), _triangles.end(), Index(0));
      if (n_pts == 2) {
        _triangles.back() = 0;
        return false;
      }
      return true;
    }
    _nodes.reserve(n_pts + 2);
    _nodes.allocate(n_pts);
    _triangles.reserve(static_cast<std::size_t>((n_pts - 3) * 1.5f));
    Index size = static_cast<Index>(n_pts);
    auto f = [](node_t &n, Index prev, Index i, Index next) {
      n.id = i;
      n.pt_id = i;
      n.next = next;
      n.prev = prev;
      n.z = 0;
      n.next_z = -1;
      n.prev_z = -1;
    };
    for (Index i = 1; i < size - 1; ++i)
      f(_nodes[i], i - 1, i, i + 1);
    f(_nodes[0], size - 1, 0, 1);
    f(_nodes[size - 1], size - 2, size - 1, 0);
    // enable z-order hashing for larger polygons
    _hashing = n_pts > 80;
    if (_hashing) {
      compute_aabb(points, 0);
    }
    return run_implementation(points, size - 1);
  }

  auto indices() const -> const tf::buffer<Index> & { return _triangles; }

  auto indices() -> tf::buffer<Index> & { return _triangles; }

  auto clear() {
    _nodes.clear();
    _triangles.clear();
    _hashing = false;
  }

private:
  class node_t {
  public:
    Index id;
    Index pt_id;
    Index next;
    Index prev;
    // z-order curve value and linked list
    int32_t z;
    Index next_z;
    Index prev_z;

    template <typename Range0, typename Range1>
    auto point(const Range0 &nodes, const Range1 &points, Index id) const
        -> tf::point<double, 2> {
      return tf::make_point_view<2>(&(points[nodes[id].pt_id][0]));
    }

    template <typename Range0>
    auto point(const Range0 &points) const -> tf::point<double, 2> {
      return tf::make_point_view<2>(&(points[pt_id][0]));
    }
  };

  template <typename Range>
  auto run_implementation(const Range &points, const Index &start_id,
                          int pass = 0) -> bool {
    // build z-order index on first pass if hashing enabled
    if (!pass && _hashing) {
      index_curve(points, start_id);
    }

    auto end_id = start_id;
    auto current_id = start_id;

    bool completed = true;

    while (_nodes[current_id].prev != _nodes[current_id].next) {
      bool ear = _hashing ? is_ear_hashed(_nodes[current_id], points)
                          : is_ear(_nodes[current_id], points);
      if (ear) {
        clip_ear(_nodes[current_id]);
        current_id = next(next(_nodes[current_id])).id;
        end_id = current_id;
        continue;
      }
      if (next(next(next(_nodes[current_id]))).id == current_id) {
        clip_ear(_nodes[current_id]);
        return false;
      }

      current_id = next(_nodes[current_id]).id;
      if (current_id == end_id) {
        const auto &v = _nodes[current_id];
        if (!pass) {
          auto pair = remove_degeneracies(points, v);
          if (pair.first) {
            completed &= run_implementation(points, pair.second, 1);
          } else {
            completed &= split_earcut(points, current_id);
          }
        } else if (pass == 1) {
          completed &= split_earcut(points, current_id);
        }
        break;
      }
    }
    return completed;
  }

  auto split_polygon(Index a_id, Index b_id) {
    Index n_id = static_cast<Index>(_nodes.size());
    auto na = _nodes[a_id];
    auto nb = _nodes[b_id];
    _nodes.push_back(na);
    _nodes.push_back(nb);
    auto &a = _nodes[a_id];
    auto &b = _nodes[b_id];

    auto &a2 = _nodes[n_id];
    a2.id = n_id;
    auto &b2 = _nodes[n_id + 1];
    b2.id = n_id + 1;
    //
    auto an_id = a.next;
    auto bp_id = b.prev;
    //
    a.next = b.id;
    b.prev = a.id;
    //
    a2.prev = b2.id;
    _nodes[an_id].prev = a2.id;
    //
    b2.next = a2.id;
    _nodes[bp_id].next = b2.id;
    //
    return n_id + 1;
  }

  template <typename Range>
  auto split_earcut(const Range &points, Index start_id, std::true_type) {
    auto run_split = [&](auto a, auto b) {
      auto c_id = split_polygon(a, b);
      bool success = run_implementation(points, a);
      success = success && run_implementation(points, c_id);
      return success;
    };
    auto a = start_id;
    std::pair<int, double> flag{-1, -std::numeric_limits<double>::max()};
    auto picked_a = a;
    auto picked_b = a;
    do {
      auto b = next(next(_nodes[a])).id;
      while (b != start_id && b != prev(_nodes[a]).id) {
        if (_nodes[a].pt_id != _nodes[b].pt_id) {
          auto diag_flag = is_valid_diagonal(points, _nodes[a], _nodes[b]);
          if (diag_flag == 7) {
            return run_split(a, b);
          } else {
            auto local_flag = std::make_pair(
                diag_flag,
                -(_nodes[a].point(points) - _nodes[b].point(points)).length2());
            if (local_flag > flag) {
              picked_a = a;
              picked_b = b;
              flag = local_flag;
            }
          }
        }
        b = next(_nodes[b]).id;
      }
      a = next(_nodes[a]).id;
    } while (a != start_id);
    run_split(picked_a, picked_b);
    return false;
  }

  template <typename Range>
  auto split_earcut(const Range &points, Index start_id, std::false_type) {
    auto run_split = [&](auto a, auto b) {
      auto c_id = split_polygon(a, b);
      bool success = run_implementation(points, a);
      success &= run_implementation(points, c_id);
      return success;
    };
    auto a = start_id;
    do {
      auto b = next(next(_nodes[a])).id;
      while (b != prev(_nodes[a]).id) {
        if (_nodes[a].pt_id != _nodes[b].pt_id &&
            is_valid_diagonal(points, _nodes[a], _nodes[b])) {
          auto c_id = split_polygon(a, b);
          bool success = run_implementation(points, a);
          success &= run_implementation(points, c_id);
          return success;
        }
        b = next(_nodes[b]).id;
      }
      a = next(_nodes[a]).id;
    } while (a != start_id);
    return false;
  }

  template <typename Range>
  auto split_earcut(const Range &points, Index start_id) {
    return split_earcut(points, start_id,
                        std::integral_constant<bool, ForceAllPoints>{});
  }

  template <typename Range> auto is_ear(const node_t &v, const Range &points) {
    if (area(points, prev(v), v, next(v)) >= 0)
      return false;

    auto current_id = next(next(v)).id;
    while (current_id != prev(v).id) {
      if (contains_point(prev(v).point(points), v.point(points),
                         next(v).point(points),
                         _nodes[current_id].point(points)) &&
          area(points, prev(_nodes[current_id]), _nodes[current_id],
               next(_nodes[current_id])) >= 0)
        return false;
      current_id = next(_nodes[current_id]).id;
    }
    return true;
  }

  auto clip_ear(node_t &v) {
    auto &pv = prev(v);
    auto &nv = next(v);

    pv.next = nv.id;
    nv.prev = pv.id;
    // update z-order linked list
    if (v.prev_z != -1)
      _nodes[v.prev_z].next_z = v.next_z;
    if (v.next_z != -1)
      _nodes[v.next_z].prev_z = v.prev_z;

    _triangles.push_back(pv.pt_id);
    _triangles.push_back(v.pt_id);
    _triangles.push_back(nv.pt_id);
  }

  auto remove_degenerate_ear(node_t &v, std::true_type) { clip_ear(v); }

  auto remove_degenerate_ear(node_t &v, std::false_type) {
    auto &pv = prev(v);
    auto &nv = next(v);

    pv.next = nv.id;
    nv.prev = pv.id;
    // update z-order linked list
    if (v.prev_z != -1)
      _nodes[v.prev_z].next_z = v.next_z;
    if (v.next_z != -1)
      _nodes[v.next_z].prev_z = v.prev_z;
  }

  auto remove_degenerate_ear(node_t &v) {
    remove_degenerate_ear(v, std::integral_constant<bool, ForceAllPoints>{});
  }

  template <typename Range>
  Index find_degenerate_ear(const Range &points, const node_t &start) const {
    Index current = start.id;

    if (start.next == start.id || _nodes[start.next].next == start.id) {
      return -1;
    }

    do {
      const auto &v = _nodes[current];
      if (v.point(points) == next(v).point(points)) {
        return current;
      } else if (v.point(points) == next(next(v)).point(points)) {
        if (next(v).point(points) == next(next(next(v))).point(points)) {
          // a 'z' in the loop: z (a) b a b c -> remove a-b-a -> z (a) a b c ->
          // remove a-a-b (next loop) -> z a b c
          // z --(a)-- b
          //         /
          //        /
          //      a -- b -- d
          return v.next;
        } else {
          // a 'shard' in the loop: z (a) b a c d -> remove a-b-a -> z (a) a b c
          // d -> remove a-a-b (next loop) -> z a b c d
          // z --(a)-- b
          //         /
          //        /
          //      a -- c -- d
          // n.b. can only do this if the shard is pointing out of the polygon.
          // i.e. b is outside z-a-c
          if (tf::classify(next(v).point(points),
                           tf::make_wedge(v.point(points),
                                          next(next(next(v))).point(points),
                                          prev(v).point(points))) ==
              tf::strict_containment::outside) {
            return v.next;
          }
        }
      }
      current = v.next;
    } while (current != start.id);

    return -1;
  }

  template <typename Range>
  auto remove_degeneracies(const Range &points, const node_t &start) {
    Index current = start.id;
    Index to_remove = -1;
    bool removed_any = false;
    while ((to_remove = find_degenerate_ear(points, _nodes[current])) != -1) {
      auto next = _nodes[to_remove].next;
      remove_degenerate_ear(_nodes[to_remove]);
      current = next;
      removed_any = true;
    }

    return std::make_pair(removed_any, current);
  }

  auto next(const node_t &node) const -> const node_t & {
    return _nodes[node.next];
  }

  auto prev(const node_t &node) const -> const node_t & {
    return _nodes[node.prev];
  }

  auto next(const node_t &node) -> node_t & { return _nodes[node.next]; }

  auto prev(const node_t &node) -> node_t & { return _nodes[node.prev]; }

  template <typename Range>
  auto area(const Range &points, const node_t &p_, const node_t &q_,
            const node_t &r_) const {
    const auto &p = p_.point(points);
    const auto &r = r_.point(points);
    const auto &q = q_.point(points);
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
  }

  auto sign(double val) { return (0.0 < val) - (val < 0.0); }

  template <typename Range>
  auto on_segment(const Range &points, const node_t &p_, const node_t &q_,
                  const node_t &r_) {
    const auto &p = p_.point(points);
    const auto &r = r_.point(points);
    const auto &q = q_.point(points);
    return q[0] <= std::max<double>(p[0], r[0]) &&
           q[0] >= std::min<double>(p[0], r[0]) &&
           q[1] <= std::max<double>(p[1], r[1]) &&
           q[1] >= std::min<double>(p[1], r[1]);
  }

  template <typename Range>
  auto intersects(const Range &points, const node_t &p1, const node_t &q1,
                  const node_t &p2, const node_t &q2) {
    int o1 = sign(area(points, p1, q1, p2));
    int o2 = sign(area(points, p1, q1, q2));
    int o3 = sign(area(points, p2, q2, p1));
    int o4 = sign(area(points, p2, q2, q1));

    if (o1 != o2 && o3 != o4)
      return true; // general case

    if (o1 == 0 && on_segment(points, p1, p2, q1))
      return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
    if (o2 == 0 && on_segment(points, p1, q2, q1))
      return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
    if (o3 == 0 && on_segment(points, p2, p1, q2))
      return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
    if (o4 == 0 && on_segment(points, p2, q1, q2))
      return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

    return false;
  }

  template <typename Range>
  auto intersects_polygon(const Range &points, const node_t &a,
                          const node_t &b) {
    Index p_id = a.id;
    do {
      const auto &p = _nodes[p_id];
      if (p.pt_id != a.pt_id && next(p).pt_id != a.pt_id &&
          p.pt_id != b.pt_id && next(p).pt_id != b.pt_id &&
          intersects(points, p, next(p), a, b))
        return true;
      p_id = next(p).id;
    } while (p_id != a.id);

    return false;
  }

  template <typename Range>
  auto locally_inside(const Range &points, const node_t &a, const node_t &b) {
    return area(points, prev(a), a, next(a)) < 0
               ? area(points, a, b, next(a)) >= 0 &&
                     area(points, a, prev(a), b) >= 0
               : area(points, a, b, prev(a)) < 0 ||
                     area(points, a, next(a), b) < 0;
  }

  template <typename Range>
  auto middle_inside(const Range &points, const node_t &a, const node_t &b) {
    auto p_id = a.id;
    bool inside = false;
    double px = (a.point(points)[0] + b.point(points)[0]) / 2;
    double py = (a.point(points)[1] + b.point(points)[1]) / 2;
    do {
      const auto &p = _nodes[p_id];
      const auto &pt_p = p.point(points);
      const auto &pt_pn = next(p).point(points);
      if (((pt_p[1] > py) != (pt_pn[1] > py)) && pt_pn[1] != pt_p[1] &&
          (px < (pt_pn[0] - pt_p[0]) * (py - pt_p[1]) / (pt_pn[1] - pt_p[1]) +
                    pt_p[0]))
        inside = !inside;
      p_id = next(p).id;
    } while (p_id != a.id);

    return inside;
  }

  template <typename Range>
  auto is_valid_diagonal(const Range &points, const node_t &a,
                         const node_t &b) {
    int is_middle_inside = middle_inside(points, a, b) << 1;
    int does_not_cause_flip =
        ((locally_inside(points, a, b) && locally_inside(points, b, a) &&
          is_middle_inside && // locally visible
          (area(points, prev(a), a, prev(b)) != 0.0 ||
           area(points, a, prev(b), b) !=
               0.0)) || // does not create opposite-facing sectors
         (a.point(points) == b.point(points) &&
          area(points, prev(a), a, next(a)) > 0 &&
          area(points, prev(b), b, next(b)) > 0));
    int does_not_intersect =
        (next(a).pt_id != b.pt_id && prev(a).pt_id != b.pt_id &&
         !intersects_polygon(points, a, b))
        << 2;
    return does_not_intersect | does_not_cause_flip | is_middle_inside;
  }

  template <typename Point>
  auto contains_point(const Point &a, const Point &b, const Point &c,
                      const Point &p) const {
    return contains_point(a[0], a[1], b[0], b[1], c[0], c[1], p[0], p[1]);
  }

  auto contains_point(double ax, double ay, double bx, double by, double cx,
                      double cy, double px, double py) const {
    return (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 &&
           (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 &&
           (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0;
  }

  // z-order hashing methods

  template <typename Range> auto compute_aabb(const Range &points, Index start_id) {
    auto current_id = start_id;
    const auto &p0 = _nodes[current_id].point(points);
    _min_x = _max_x = p0[0];
    _min_y = _max_y = p0[1];
    current_id = _nodes[current_id].next;
    while (current_id != start_id) {
      const auto &p = _nodes[current_id].point(points);
      _min_x = std::min(_min_x, p[0]);
      _max_x = std::max(_max_x, p[0]);
      _min_y = std::min(_min_y, p[1]);
      _max_y = std::max(_max_y, p[1]);
      current_id = _nodes[current_id].next;
    }
    _inv_size = std::max(_max_x - _min_x, _max_y - _min_y);
    _inv_size = _inv_size != 0.0 ? (32767.0 / _inv_size) : 0.0;
  }

  auto z_order(double x, double y) const -> int32_t {
    // transform coords into non-negative 15-bit integer range
    int32_t ix = static_cast<int32_t>((x - _min_x) * _inv_size);
    int32_t iy = static_cast<int32_t>((y - _min_y) * _inv_size);
    // interleave bits using magic numbers (Morton code)
    ix = (ix | (ix << 8)) & 0x00FF00FF;
    ix = (ix | (ix << 4)) & 0x0F0F0F0F;
    ix = (ix | (ix << 2)) & 0x33333333;
    ix = (ix | (ix << 1)) & 0x55555555;

    iy = (iy | (iy << 8)) & 0x00FF00FF;
    iy = (iy | (iy << 4)) & 0x0F0F0F0F;
    iy = (iy | (iy << 2)) & 0x33333333;
    iy = (iy | (iy << 1)) & 0x55555555;

    return ix | (iy << 1);
  }

  template <typename Range> auto index_curve(const Range &points, Index start_id) {
    auto current_id = start_id;
    do {
      auto &node = _nodes[current_id];
      if (node.z == 0) {
        const auto &p = node.point(points);
        node.z = z_order(p[0], p[1]);
      }
      node.prev_z = node.prev;
      node.next_z = node.next;
      current_id = node.next;
    } while (current_id != start_id);
    // break the circular z-list into linear for sorting
    _nodes[_nodes[start_id].prev_z].next_z = -1;
    _nodes[start_id].prev_z = -1;
    sort_linked_z(start_id);
  }

  // Simon Tatham's linked list merge sort algorithm
  auto sort_linked_z(Index list_id) -> Index {
    Index p_id, q_id, e_id, tail_id;
    int i, num_merges, p_size, q_size;
    int in_size = 1;

    for (;;) {
      p_id = list_id;
      list_id = -1;
      tail_id = -1;
      num_merges = 0;

      while (p_id != -1) {
        num_merges++;
        q_id = p_id;
        p_size = 0;
        for (i = 0; i < in_size; ++i) {
          p_size++;
          q_id = _nodes[q_id].next_z;
          if (q_id == -1)
            break;
        }
        q_size = in_size;

        while (p_size > 0 || (q_size > 0 && q_id != -1)) {
          if (p_size == 0) {
            e_id = q_id;
            q_id = _nodes[q_id].next_z;
            q_size--;
          } else if (q_size == 0 || q_id == -1) {
            e_id = p_id;
            p_id = _nodes[p_id].next_z;
            p_size--;
          } else if (_nodes[p_id].z <= _nodes[q_id].z) {
            e_id = p_id;
            p_id = _nodes[p_id].next_z;
            p_size--;
          } else {
            e_id = q_id;
            q_id = _nodes[q_id].next_z;
            q_size--;
          }

          if (tail_id != -1)
            _nodes[tail_id].next_z = e_id;
          else
            list_id = e_id;

          _nodes[e_id].prev_z = tail_id;
          tail_id = e_id;
        }
        p_id = q_id;
      }
      _nodes[tail_id].next_z = -1;

      if (num_merges <= 1)
        return list_id;

      in_size *= 2;
    }
  }

  template <typename Range>
  auto is_ear_hashed(const node_t &v, const Range &points) {
    const auto &pv = prev(v);
    const auto &nv = next(v);
    if (area(points, pv, v, nv) >= 0)
      return false;

    // triangle coordinates
    const auto &a = pv.point(points);
    const auto &b = v.point(points);
    const auto &c = nv.point(points);

    // triangle bounding box
    double min_tx = std::min({a[0], b[0], c[0]});
    double min_ty = std::min({a[1], b[1], c[1]});
    double max_tx = std::max({a[0], b[0], c[0]});
    double max_ty = std::max({a[1], b[1], c[1]});

    // z-order range for the triangle bbox
    int32_t min_z = z_order(min_tx, min_ty);
    int32_t max_z = z_order(max_tx, max_ty);

    // search forward in z-order
    auto p_id = v.next_z;
    while (p_id != -1 && _nodes[p_id].z <= max_z) {
      const auto &p = _nodes[p_id];
      if (p.id != pv.id && p.id != nv.id &&
          contains_point(a, b, c, p.point(points)) &&
          area(points, prev(p), p, next(p)) >= 0)
        return false;
      p_id = p.next_z;
    }

    // search backward in z-order
    p_id = v.prev_z;
    while (p_id != -1 && _nodes[p_id].z >= min_z) {
      const auto &p = _nodes[p_id];
      if (p.id != pv.id && p.id != nv.id &&
          contains_point(a, b, c, p.point(points)) &&
          area(points, prev(p), p, next(p)) >= 0)
        return false;
      p_id = p.prev_z;
    }

    return true;
  }

  tf::buffer<node_t> _nodes;
  tf::buffer<Index> _triangles;
  // z-order hashing state
  bool _hashing = false;
  double _min_x = 0, _max_x = 0;
  double _min_y = 0, _max_y = 0;
  double _inv_size = 0;
};
} // namespace tf::geom
