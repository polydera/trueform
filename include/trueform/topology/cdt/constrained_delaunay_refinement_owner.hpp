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
#include "../../core/constants.hpp"
#include "../../core/point.hpp"
#include "../../exact/incircle.hpp"
#include "../../exact/orient2d.hpp"
#include "../cdt_constraint_split.hpp"
#include "../constrained_delaunay_triangulator.hpp"
#include "./constrained_delaunay_full_span_alias.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

/// Authoritative flat state for constrained Delaunay refinement. Free
/// operations in this directory mutate it phase by phase.
template <typename Index, typename Coord, typename Int>
struct constrained_delaunay_refinement_owner {
  using index_type = Index;
  using coord_type = Coord;
  using int_type = Int;
  using cdt_type = tf::constrained_delaunay_triangulator<Index, Coord, Int>;
  using param_type = typename cdt_type::param_t;
  using constraint_provenance = typename cdt_type::constraint_owner_t;
  using constraint_collision = typename cdt_type::constraint_collision;

  static constexpr int max_split_depth = 6;
  static constexpr Index none = Index(-1);
  // A sub-edge span is stated on the seed CDT's parameter scale, so a
  // refined edge and a recovered one name the same interval.
  static constexpr int crossing_param_bits = cdt_type::k_crossing_param_bits;

  struct triangle {
    Index v[3];
    Index n[3];   // neighbor across edge v[e] -> v[e+1]
    Index seg[3]; // constrained sub-edge id, none when unconstrained
    std::uint32_t stamp;
  };

  // A constrained sub-edge: interval [m/2^d, (m+1)/2^d] of input edge
  // `origin`, with `lo_vertex` the endpoint on the m/2^d side.
  struct sub_edge {
    Index origin;
    Index lo_vertex;
    std::uint8_t m;
    std::uint8_t d;
  };

  struct split_request {
    Index f;
    Index e;
    std::uint32_t stamp;
  };

  // Diametral-circle encroachment: p sees the segment (u, v) at an obtuse
  // angle (Ruppert).
  auto encroaches(const std::array<double, 2> &pu,
                  const std::array<double, 2> &pv,
                  const std::array<double, 2> &pp) const -> bool {
    double ux = pp[0] - pu[0], uy = pp[1] - pu[1];
    double vx = pp[0] - pv[0], vy = pp[1] - pv[1];
    return ux * vx + uy * vy < 0;
  }

  auto orient(Index a, Index b, Index c) const -> int {
    return tf::exact::orient2d_sign(_ip[a], _ip[b], _ip[c]);
  }

  auto incircle(Index a, Index b, Index c, Index d) const -> int {
    return tf::exact::incircle_sign(_ip[a], _ip[b], _ip[c], _ip[d]);
  }

  auto quality(Index f) const -> double {
    const auto &a = _dp[std::size_t(_t[f].v[0])];
    const auto &b = _dp[std::size_t(_t[f].v[1])];
    const auto &c = _dp[std::size_t(_t[f].v[2])];
    double ex = b[0] - a[0], ey = b[1] - a[1];
    double fx = c[0] - a[0], fy = c[1] - a[1];
    double area2 = std::abs(ex * fy - ey * fx);
    double e2 = ex * ex + ey * ey;
    double f2 = fx * fx + fy * fy;
    double g2 = (fx - ex) * (fx - ex) + (fy - ey) * (fy - ey);
    double m = std::max({e2, f2, g2});
    return m > 0 ? tf::two_over_sqrt_3<double> * area2 / m : 1.0;
  }

  auto constraint_connected(Index a, Index b) const -> bool {
    return _con_nbr[std::size_t(a)][0] == b ||
           _con_nbr[std::size_t(a)][1] == b ||
           _con_nbr[std::size_t(b)][0] == a ||
           _con_nbr[std::size_t(b)][1] == a;
  }

  auto edge_back(Index f, Index from) const -> int {
    for (int e = 0; e < 3; ++e)
      if (_t[f].n[e] == from)
        return e;
    return -1;
  }

  auto sub_depth(Index f, int e) const -> int {
    auto s = _t[f].seg[e];
    return s == none ? 0 : int(_segments[std::size_t(s)].d);
  }

  auto splittable(Index f, int e) const -> bool {
    return _split_encroached && sub_depth(f, e) < max_split_depth;
  }

  auto constrained(Index f, int e) const -> bool {
    return _t[f].seg[e] != none;
  }

  cdt_type _cdt;
  // Per-build scratch: reused across builds, never freed.
  tf::buffer<constrained_delaunay_full_span_alias<Index>> _constraint_aliases;
  tf::buffer<std::array<Index, 2>> _constraint_alias_blocks;
  tf::buffer<tf::point<Int, 2>> _ip;
  tf::buffer<std::array<double, 2>> _dp;
  tf::buffer<triangle> _t;
  tf::buffer<Index> _label;
  tf::buffer<sub_edge> _segments;
  tf::buffer<std::array<Index, 2>> _con_nbr;
  tf::buffer<std::array<Index, 2>> _queue;
  tf::buffer<Index> _cavity;
  tf::buffer<Index> _lawson;
  tf::buffer<split_request> _pending;
  tf::buffer<split_request> _generation;
  tf::buffer<tf::cdt_constraint_split<Index>> _splits;
  tf::buffer<Index> _split_offsets;
  tf::buffer<Index> _representative_split_offsets;
  tf::buffer<tf::cdt_constraint_split<Index>> _expanded_splits;
  tf::buffer<Index> _constraint_split_parent;
  tf::buffer<char> _constraint_split_reversed;
  tf::buffer<char> _steiner_point_flags;
  double _scale = 1;
  double _offx = 0;
  double _offy = 0;
  Index _n_input_points = 0;
  Index _n_input_edges = 0;
  bool _split_encroached = true;
  double _min_quality = 0.0;
  bool _ok = false;
};

} // namespace tf::topology::cdt
