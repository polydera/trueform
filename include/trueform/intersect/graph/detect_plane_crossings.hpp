/*
 * Copyright (c) 2026 XLAB
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

#include "../../core/aabb.hpp"
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/intersects.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/segments.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/edge_parameter.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/parameter_line_sign2.hpp"
#include "../../exact/parameter_plane_sign.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/search.hpp"
#include "../../spatial/search_self.hpp"
#include "../../spatial/tree_config.hpp"
#include "../polygon_intersections.hpp"
#include "./face_descriptor.hpp"
#include "./face_id.hpp"
#include "./plane_census.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_face_support.hpp"
#include "./plane_graph.hpp"
#include "./plane_identity_names.hpp"
#include "./plane_pair_carrier.hpp"
#include "./plane_point_generator.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::intersect::graph {

/// A line inside a plane carrier, as its GENERATORS state it.
///
/// Two forms, both exact for every sign: the trace of a transversal face
/// plane (three lattice points), or the line through two lattice points.
/// The two-point form additionally names a lattice FRAME, and that frame
/// is what bounds the width of a crossing's exact fraction — a position
/// stated on it is one orientation determinant over its difference,
/// degree three, the T2 rung. It is preferred wherever it exists.
template <typename Index, typename Int> struct plane_line {
  tf::exact::pt3<Int> a{}, b{}, c{};
  tf::exact::pt3<Int> ga{}, gb{};
  Index flat_ga = Index(-1), flat_gb = Index(-1);
  bool is_plane = false;
  bool has_frame = false;
  bool valid = false;
};

/// The direction of a line, sign-only: the lattice difference of a
/// frame, or the degree-four carrier of a transversal pair carried
/// through its Lagrange decomposition. A frame direction enters the same
/// carrier form with `a = 1, b1 = w, b = 0`, so one order predicate
/// serves both.
template <typename Int>
auto plane_frame_direction(
    const std::array<typename tf::exact::meta<Int>::T1, 3> &w)
    -> plane_pair_carrier<Int> {
  plane_pair_carrier<Int> carrier;
  carrier.b1 = w;
  carrier.b2 = {typename tf::exact::meta<Int>::T1(0),
                typename tf::exact::meta<Int>::T1(0),
                typename tf::exact::meta<Int>::T1(0)};
  carrier.a = typename tf::exact::meta<Int>::T2(1);
  carrier.b = typename tf::exact::meta<Int>::T2(0);
  carrier.valid = true;
  return carrier;
}

/// Sign of a generator-stated point against a line: zero is an exact
/// incidence. A parameterized point never materializes — the parameter
/// enters the determinant.
template <typename Index, typename Int>
auto plane_line_sign(const plane_line<Index, Int> &line,
                     const plane_point_generator<Index, Int> &generator,
                     int ax0, int ax1) -> int {
  using T2 = typename tf::exact::meta<Int>::T2;
  using pt2_t = tf::exact::pt2<Int>;
  if (line.is_plane) {
    if (!generator.on_edge) {
      const auto volume =
          tf::exact::orient3d_value(line.a, line.b, line.c, generator.u);
      return volume > T2(0) ? 1 : (volume < T2(0) ? -1 : 0);
    }
    return tf::exact::parameter_plane_sign<Int>(
        line.a, line.b, line.c, generator.u, generator.v, generator.t);
  }
  const pt2_t ga{line.ga[ax0], line.ga[ax1]}, gb{line.gb[ax0], line.gb[ax1]};
  if (!generator.on_edge) {
    const pt2_t q{generator.u[ax0], generator.u[ax1]};
    const auto o = tf::exact::orient2d(ga, gb, q);
    return o > T2(0) ? 1 : (o < T2(0) ? -1 : 0);
  }
  const pt2_t u{generator.u[ax0], generator.u[ax1]},
      v{generator.v[ax0], generator.v[ax1]};
  return tf::exact::parameter_line_sign2<Int>(ga, gb, u, v, generator.t);
}

/// Exact equality of two generator-stated points, where the generators
/// decide it: two lattice points compare coordinates, two points on one
/// carrier compare parameters. Across distinct carriers the question
/// needs a rung above T2 and the answer is `false` — never a guess.
template <typename Index, typename Int>
auto plane_generators_equal(const plane_point_generator<Index, Int> &x,
                            const plane_point_generator<Index, Int> &y)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (!x.on_edge && !y.on_edge)
    return x.u[0] == y.u[0] && x.u[1] == y.u[1] && x.u[2] == y.u[2];
  if (x.on_edge && y.on_edge)
    return x.carrier_u == y.carrier_u && x.carrier_v == y.carrier_v &&
           tf::exact::compare_parameter<Int>(x.t, y.t) == 0;
  // a lattice point against a parameterized one: the carrier's own
  // parameter of the lattice point is degree two over degree two, so
  // the comparison is one determinant sign and never leaves T2
  const auto &parameterized = x.on_edge ? x : y;
  const auto &lattice = x.on_edge ? y : x;
  const std::array<T1, 3> w{T1(parameterized.v[0]) - parameterized.u[0],
                            T1(parameterized.v[1]) - parameterized.u[1],
                            T1(parameterized.v[2]) - parameterized.u[2]};
  const std::array<T1, 3> d{T1(lattice.u[0]) - parameterized.u[0],
                            T1(lattice.u[1]) - parameterized.u[1],
                            T1(lattice.u[2]) - parameterized.u[2]};
  for (int k = 0; k < 3; ++k) {
    const int a = (k + 1) % 3, b = (k + 2) % 3;
    if (!(T2(w[std::size_t(a)]) * T2(d[std::size_t(b)]) -
              T2(w[std::size_t(b)]) * T2(d[std::size_t(a)]) ==
          T2(0)))
      return false;
  }
  const T2 num =
      T2(w[0]) * T2(d[0]) + T2(w[1]) * T2(d[1]) + T2(w[2]) * T2(d[2]);
  const T2 den =
      T2(w[0]) * T2(w[0]) + T2(w[1]) * T2(w[1]) + T2(w[2]) * T2(w[2]);
  return tf::exact::compare_parameter<Int>({num, den}, parameterized.t) == 0;
}

/// A line whose two endpoints share one original-edge carrier IS that
/// carrier's line: two distinct points determine it. One exact sign
/// validates the candidate, and it is paid only where a fraction is
/// about to be stated on it.
template <typename Index, typename Int>
auto promote_plane_line_frame(const plane_point_generator<Index, Int> &g0,
                              const plane_point_generator<Index, Int> &g1,
                              int ax0, int ax1, plane_line<Index, Int> &line)
    -> bool {
  auto candidate = [&](const plane_point_generator<Index, Int> &from,
                       const plane_point_generator<Index, Int> &other) -> bool {
    if (!from.on_edge)
      return false;
    plane_line<Index, Int> probe;
    probe.ga = from.u;
    probe.gb = from.v;
    probe.flat_ga = from.carrier_u;
    probe.flat_gb = from.carrier_v;
    if (probe.ga[ax0] == probe.gb[ax0] && probe.ga[ax1] == probe.gb[ax1])
      return false;
    if (plane_line_sign<Index, Int>(probe, other, ax0, ax1) != 0)
      return false;
    line.ga = probe.ga;
    line.gb = probe.gb;
    line.flat_ga = probe.flat_ga;
    line.flat_gb = probe.flat_gb;
    return true;
  };
  return candidate(g0, g1) || candidate(g1, g0);
}

/// The line of a canonical edge. The frame form is preferred: it is
/// the same line (a carrier validated to carry BOTH endpoints, or the
/// two lattice endpoints themselves) and it is the only form that
/// names a lattice frame for a crossing's fraction.
/// The line of a canonical edge, read off what the definition already
/// states: a boundary instance names the ORIGINAL EDGE it lies on, so
/// its lattice frame costs no predicate at all; an interior cut is the
/// trace of its transversal partner's plane. Nothing here evaluates an
/// exact predicate — the frame a crossing needs for its fraction is
/// promoted at the crossing, which happens hundreds of times, not per
/// edge, which happens hundreds of thousands of times.
template <typename Index, typename Int, typename IsMember, typename FacePoints,
          typename EdgeFrame>
auto build_plane_line(const plane_point_generator<Index, Int> &g0,
                      const plane_point_generator<Index, Int> &g1, Index flat_0,
                      Index flat_1, const plane_edge_def<Index> &def,
                      const IsMember &is_member, const FacePoints &face_points,
                      const EdgeFrame &edge_frame, int ax0, int ax1,
                      plane_line<Index, Int> &line) -> void {
  auto accept = [&](const tf::exact::pt3<Int> &ga,
                    const tf::exact::pt3<Int> &gb, Index fa, Index fb) -> bool {
    if (ga[ax0] == gb[ax0] && ga[ax1] == gb[ax1])
      return false;
    line.ga = ga;
    line.gb = gb;
    line.flat_ga = fa;
    line.flat_gb = fb;
    line.is_plane = false;
    line.has_frame = true;
    line.valid = true;
    return true;
  };
  // the generators may already name the frame: two points of one
  // carrier, or a carrier and one of its own endpoints
  if (g0.on_edge && g1.on_edge && g0.carrier_u == g1.carrier_u &&
      g0.carrier_v == g1.carrier_v &&
      accept(g0.u, g0.v, g0.carrier_u, g0.carrier_v))
    return;
  if (g0.on_edge != g1.on_edge) {
    const auto &edge = g0.on_edge ? g0 : g1;
    const auto vertex = g0.on_edge ? flat_1 : flat_0;
    if ((vertex == edge.carrier_u || vertex == edge.carrier_v) &&
        accept(edge.u, edge.v, edge.carrier_u, edge.carrier_v))
      return;
  }
  if (!g0.on_edge && !g1.on_edge) {
    if (flat_1 < flat_0 ? accept(g1.u, g0.u, flat_1, flat_0)
                        : accept(g0.u, g1.u, flat_0, flat_1))
      return;
  }

  if (def.side >= 0) {
    tf::exact::pt3<Int> ga, gb;
    Index fa = Index(-1), fb = Index(-1);
    if (edge_frame(def, ga, gb, fa, fb) && accept(ga, gb, fa, fb))
      return;
  }
  if (!is_member(def.tag_other, def.object_other)) {
    line.is_plane = true;
    line.has_frame = false;
    face_points(def.tag_other, def.object_other, line.a, line.b, line.c);
    line.valid = true;
    return;
  }
  line.has_frame = promote_plane_line_frame(g0, g1, ax0, ax1, line);
  line.is_plane = false;
  line.valid = line.has_frame;
}

/// A line's direction, sign-only: the lattice difference of its frame,
/// or the degree-four carrier of the plane pair that cuts it out, kept
/// in its Lagrange decomposition so no operand leaves the T2 rung.
///
/// A carrier whose support is collinear HAS no plane — it is a line, and
/// there is no pair to intersect. Every edge stated on it lies along it,
/// so the support's own direction is the answer, on the same rung.
template <typename Index, typename Int>
auto plane_line_direction(const plane_line<Index, Int> &line,
                          const tf::exact::pt3<Int> &own_a,
                          const tf::exact::pt3<Int> &own_b,
                          const tf::exact::pt3<Int> &own_c)
    -> plane_pair_carrier<Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (line.has_frame) {
    const std::array<T1, 3> w{T1(line.gb[0]) - line.ga[0],
                              T1(line.gb[1]) - line.ga[1],
                              T1(line.gb[2]) - line.ga[2]};
    return plane_frame_direction<Int>(w);
  }
  auto basis = [](const tf::exact::pt3<Int> &p, const tf::exact::pt3<Int> &q) {
    return std::array<T1, 3>{T1(q[0]) - p[0], T1(q[1]) - p[1], T1(q[2]) - p[2]};
  };
  const auto a1 = basis(own_a, own_b);
  const auto a2 = basis(own_a, own_c);
  const auto b1 = basis(line.a, line.b);
  const auto b2 = basis(line.a, line.c);
  const auto support_direction = [](const std::array<T1, 3> &e0,
                                    const std::array<T1, 3> &e1) {
    const auto &w =
        e0[0] != T1(0) || e0[1] != T1(0) || e0[2] != T1(0) ? e0 : e1;
    return w[0] == T1(0) && w[1] == T1(0) && w[2] == T1(0)
               ? plane_pair_carrier<Int>{}
               : plane_frame_direction<Int>(w);
  };
  const auto collapsed = [](const std::array<T1, 3> &e0,
                            const std::array<T1, 3> &e1) {
    const auto normal = plane_carrier_cross<Int>(e0, e1);
    return normal[0] == T2(0) && normal[1] == T2(0) && normal[2] == T2(0);
  };
  if (collapsed(a1, a2))
    return support_direction(a1, a2);
  if (collapsed(b1, b2))
    return support_direction(b1, b2);
  auto carrier = make_plane_pair_carrier<Int>(a1, a2, b1, b2);
  if (carrier.a == T2(0) && carrier.b == T2(0))
    carrier.valid = false;
  return carrier;
}

/// Two edges on one carrier line cannot cross in their interiors, so
/// no EE machinery runs: the only facts are an endpoint inside the
/// other's span, and two endpoints at one position.
template <typename Index, typename Int, typename Local, typename Sink,
          typename EmitLanding, typename EndpointName>
auto plane_collinear_pair(const Local &local, Sink &sink, std::size_t i,
                          std::size_t j, const plane_line<Index, Int> &la,
                          const plane_line<Index, Int> &lb,
                          const tf::exact::pt3<Int> &own_a,
                          const tf::exact::pt3<Int> &own_b,
                          const tf::exact::pt3<Int> &own_c,
                          const EmitLanding &emit_landing,
                          const EndpointName &endpoint_name) -> void {
  const auto dir_a = plane_line_direction(la, own_a, own_b, own_c);
  const auto dir_b = plane_line_direction(lb, own_a, own_b, own_c);
  const auto &a0 = local.gens[2 * i];
  const auto &a1 = local.gens[2 * i + 1];
  const auto &b0 = local.gens[2 * j];
  const auto &b1 = local.gens[2 * j + 1];
  // both points are on the line, so a zero order IS coincidence with
  // an endpoint — the one case that is not an interior landing
  auto strictly_inside = [](const plane_pair_carrier<Int> &dir,
                            const plane_point_generator<Index, Int> &q0,
                            const plane_point_generator<Index, Int> &q1,
                            const plane_point_generator<Index, Int> &q) -> int {
    const int lo = plane_edge_carrier_sign<Index, Int>(dir, q0, q);
    const int hi = plane_edge_carrier_sign<Index, Int>(dir, q, q1);
    if (lo == 0 || hi == 0)
      return -1;
    return lo == hi ? 1 : -1;
  };
  if (!dir_a.valid || !dir_b.valid)
    ++sink.census.undecided_order;
  const auto canon_a = local.cedges[i].canon;
  const auto canon_b = local.cedges[j].canon;
  if (dir_b.valid) {
    if (strictly_inside(dir_b, b0, b1, a0) > 0)
      emit_landing(canon_b, canon_a, endpoint_name(i, 0), sink);
    if (strictly_inside(dir_b, b0, b1, a1) > 0)
      emit_landing(canon_b, canon_a, endpoint_name(i, 1), sink);
  }
  if (dir_a.valid) {
    if (strictly_inside(dir_a, a0, a1, b0) > 0)
      emit_landing(canon_a, canon_b, endpoint_name(j, 0), sink);
    if (strictly_inside(dir_a, a0, a1, b1) > 0)
      emit_landing(canon_a, canon_b, endpoint_name(j, 1), sink);
  }
}

/// THE DETECTION — every junction a plane's own edges state, named by
/// the original features that define it.
///
/// It runs over CANONICAL edges, not instances: a plane's edge block
/// arrives canon-grouped, so the unique ids are a run walk, coincident
/// sides are intersected once, and the same-canon pair filter is
/// unconstructible. A crossing's POSITION is an exact fraction on a
/// lattice frame — never a materialized created coordinate — so no
/// decision here reads a rounded input.
template <typename Index, typename RealType, typename Int, typename GetPoint,
          typename GetMeshPoint, typename ApplyToFace, typename VertexOffsets>
auto detect_plane_crossings(
    const plane_graph<Index, Int> &g,
    const tf::polygon_intersections<Index, RealType, Int> &ibp,
    const GetPoint &get_point, const GetMeshPoint &get_mesh_point,
    const ApplyToFace &apply_to_face, const VertexOffsets &vertex_offsets,
    bool resolve_self_contours,
    tf::buffer<plane_name_statement<Index>> &statements, plane_census &census)
    -> void {
  using fid_t = face_id<Index>;
  using def_t = plane_edge_def<Index>;
  using gen_t = plane_point_generator<Index, Int>;
  using line_t = plane_line<Index, Int>;
  using pt3_t = tf::exact::pt3<Int>;
  using T1 = typename tf::exact::meta<Int>::T1;
  const auto defs = g.edge_defs();
  const auto descriptors = g.descriptors();

  // one canonical edge of one plane: the instance whose line states it,
  // the contour its instances fold to, and whether it is a line at all
  struct cedge_t {
    Index canon;
    Index def;
    std::int16_t contour;
    char valid;
  };

  /// A wide carrier's own chunk of the second grain: the statements it
  /// states and the census it counts, aggregated in block order.
  struct wide_t {
    tf::buffer<plane_name_statement<Index>> statements;
    plane_census census;
  };

  struct local_t {
    wide_t out;
    tf::buffer<cedge_t> cedges;
    tf::buffer<gen_t> gens;
    tf::buffer<line_t> lines;
    tf::buffer<pt3_t> spts;
    tf::buffer<fid_t> members;
    tf::points_buffer<Int, 2> pts_2d;
    tf::buffer<int> seg_ids;
    tf::aabb_tree<int, Int, 2> tree;
  };

  auto task = [&](auto &&range, local_t &local) {
    /// THE THRESHOLD, from the measurement and not from convention. A
    /// carrier's crossings cost about `n log n`, and the second grain costs
    /// a nested partition, so it earns its keep only where one carrier is a
    /// visible share of the pass. On the corpus at tolerance the tail is
    /// unmistakable: a pair's largest carrier holds 44-65% of its edges
    /// while 95% of carriers hold fewer than 256, so the average carrier
    /// keeps the single-grain walk untouched and the second grain engages
    /// for the handful that are the pass.
    constexpr std::size_t wide_carrier_edges = 256;

    auto get_point_f = get_point;
    auto get_mesh_point_f = get_mesh_point;
    auto apply_to_face_f = apply_to_face;

    auto is_member = [&](std::int16_t tag, Index object) -> bool {
      return std::binary_search(local.members.begin(), local.members.end(),
                                fid_t{short(tag), object});
    };
    auto group_of = [&](std::int16_t tag, Index object) -> Index {
      const auto at = std::lower_bound(
          descriptors.begin(), descriptors.end(),
          face_descriptor<Index>{Index(tag), object},
          [](const face_descriptor<Index> &x, const face_descriptor<Index> &y) {
            return std::tie(x.tag, x.object) < std::tie(y.tag, y.object);
          });
      if (at == descriptors.end() || at->tag != Index(tag) ||
          at->object != object)
        return Index(-1);
      return Index(at - descriptors.begin());
    };
    // the lattice frame a boundary instance already names: the
    // ORIGINAL EDGE it lies on, in the flat ids' own order so every
    // face that states the edge states the same frame
    auto flat_point = [&](Index flat) {
      const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
      return get_mesh_point_f(int(tag),
                              flat - vertex_offsets[std::size_t(tag)]);
    };
    // a frame is an IBP CARRIER: its canonical flat ids name it and
    // its canonical positions are the ones every blend on it uses, so
    // a fraction stated here lands where the intersection points of
    // the same carrier landed
    auto edge_frame = [&](const def_t &def, pt3_t &ga, pt3_t &gb, Index &fa,
                          Index &fb) -> bool {
      const auto tag = std::int16_t(descriptors[std::size_t(def.face)].tag);
      const auto object = descriptors[std::size_t(def.face)].object;
      bool named = false;
      apply_to_face_f(tag, object, [&](const auto &face) {
        const auto n = face.size();
        if (std::size_t(def.side) >= n)
          return;
        fa = ibp.canonical_vertex(tag, Index(face[std::size_t(def.side)]));
        fb = ibp.canonical_vertex(tag,
                                  Index(face[(std::size_t(def.side) + 1) % n]));
        named = true;
      });
      if (!named)
        return false;
      if (fb < fa)
        std::swap(fa, fb);
      ga = flat_point(fa);
      gb = flat_point(fb);
      return true;
    };
    auto face_points = [&](std::int16_t tag, Index object, pt3_t &a, pt3_t &b,
                           pt3_t &c) {
      apply_to_face_f(tag, object, [&](const auto &face) {
        plane_face_support<Int>(
            face,
            [&](std::size_t k) {
              return get_mesh_point_f(int(tag), Index(face[k]));
            },
            a, b, c);
      });
    };
    // the generator of an endpoint, in the ibp's own currency: a
    // lattice vertex, or an exact fraction of an original edge
    auto generator_of = [&](std::int16_t tag, Index id, Index &flat) -> gen_t {
      gen_t generator;
      if (tag >= 0) {
        generator.u = get_mesh_point_f(int(tag), id);
        flat = ibp.canonical_vertex(tag, id);
        return generator;
      }
      if (id < ibp.n_vertex_points()) {
        const auto &anchor = ibp.vertex_anchor(id);
        generator.u = get_mesh_point_f(int(anchor.tag), anchor.vid);
        flat = vertex_offsets[std::size_t(anchor.tag)] + anchor.vid;
        return generator;
      }
      const auto &home = ibp.home_edge(id);
      const auto tag_u = tf::exact::tag_of_flat_vertex(vertex_offsets, home.u);
      const auto tag_v = tf::exact::tag_of_flat_vertex(vertex_offsets, home.v);
      generator.u = get_mesh_point_f(
          int(tag_u), home.u - vertex_offsets[std::size_t(tag_u)]);
      generator.v = get_mesh_point_f(
          int(tag_v), home.v - vertex_offsets[std::size_t(tag_v)]);
      generator.t = ibp.exact_parameter(id);
      generator.carrier_u = home.u;
      generator.carrier_v = home.v;
      generator.on_edge = true;
      flat = Index(-1);
      return generator;
    };

    for (auto &&[p, faces_of_plane] : range) {
      // a zero-normal plane is a LINE: its member collapsed to a
      // segment, so every coincidence lives inside one contour and the
      // contour gates below do not apply — the walk itself is the
      // resolution, so it runs for such a plane whatever the gates say
      const auto &plane_frame = g.frame(Index(p));
      using frame_T2 = std::decay_t<decltype(plane_frame.plane_n[0])>;
      const bool line_plane = plane_frame.plane_n[0] == frame_T2(0) &&
                              plane_frame.plane_n[1] == frame_T2(0) &&
                              plane_frame.plane_n[2] == frame_T2(0);
      std::int16_t contour = -1;
      bool multi = false;
      for (const auto f : faces_of_plane) {
        const auto c = g.face_contour()[std::size_t(f)];
        if (c == std::int16_t(-1))
          continue;
        if (c == std::int16_t(-2) ||
            (contour != std::int16_t(-1) && c != contour)) {
          multi = true;
          break;
        }
        contour = c;
      }
      if (!line_plane && !resolve_self_contours && !multi &&
          faces_of_plane.size() < 2)
        continue;
      const auto plane = Index(p);
      const auto block = g.plane_edges(plane);
      if (block.size() < 2)
        continue;

      local.members.clear();
      tf::core::reallocate(local.members, faces_of_plane.size());
      auto *member = local.members.begin();
      for (const auto f : faces_of_plane)
        *member++ = fid_t{short(descriptors[std::size_t(f)].tag),
                          Index(descriptors[std::size_t(f)].object)};
      std::sort(local.members.begin(), local.members.end());

      const auto &frame = g.frame(plane);
      const int ax0 = frame.ax0, ax1 = frame.ax1;
      pt3_t own_a, own_b, own_c;
      face_points(std::int16_t(local.members[0].tag), local.members[0].object,
                  own_a, own_b, own_c);

      // the plane's canonical edges: the block is canon-grouped, so a
      // run walk IS the unique-id fold, and an instance's attributes
      // fold into the canonical edge's
      local.cedges.clear();
      for (std::size_t i = 0; i < block.size();) {
        const auto canon = defs[std::size_t(block[i])].id;
        std::size_t j = i;
        std::int16_t fold = -1;
        Index transversal = Index(-1);
        while (j < block.size() && defs[std::size_t(block[j])].id == canon) {
          const auto &def = defs[std::size_t(block[j])];
          if (fold == std::int16_t(-1))
            fold = def.tag_other;
          else if (fold != def.tag_other)
            fold = -2;
          if (transversal == Index(-1) &&
              !is_member(def.tag_other, def.object_other))
            transversal = block[j];
          ++j;
        }
        local.cedges.push_back(
            {canon, transversal == Index(-1) ? block[i] : transversal, fold,
             1});
        i = j;
      }
      const auto n = local.cedges.size();
      if (n < 2)
        continue;
      local.out.census.canonical_edges += n;

      // an endpoint that lands on another edge NAMES itself: the
      // junction IS that existing point, so ibp unification is
      // automatic and nothing is minted
      auto endpoint_name = [&](std::size_t i, int side) -> plane_name<Index> {
        const auto &def = defs[std::size_t(local.cedges[i].def)];
        const auto tag = side == 0 ? def.point_tag_0 : def.point_tag_1;
        const auto id = side == 0 ? def.point_0 : def.point_1;
        return tag < 0
                   ? plane_point_name<Index>(id)
                   : plane_vertex_name<Index>(ibp.canonical_vertex(tag, id));
      };
      // THE SINK a pair test states into. The kernel below is one kernel;
      // only where its statements land differs between the two grains, so
      // neither grain can drift from the other's verdict.
      auto emit_landing = [&](Index root, Index partner,
                              const plane_name<Index> &name, wide_t &sink) {
        sink.statements.push_back({name, root, partner, plane});
        ++sink.census.crossings_ve;
        if (name[0] == Index(0))
          ++sink.census.names_point;
        else
          ++sink.census.names_vertex;
      };

      local.gens.clear();
      local.lines.clear();
      local.spts.clear();
      tf::core::reallocate(local.gens, 2 * n);
      tf::core::reallocate(local.lines, n);
      tf::core::reallocate(local.spts, 2 * n);
      for (std::size_t i = 0; i < n; ++i) {
        const auto &def = defs[std::size_t(local.cedges[i].def)];
        auto &g0 = local.gens[2 * i];
        auto &g1 = local.gens[2 * i + 1];
        Index flat_0 = Index(-1), flat_1 = Index(-1);
        g0 = generator_of(def.point_tag_0, def.point_0, flat_0);
        g1 = generator_of(def.point_tag_1, def.point_1, flat_1);
        local.spts[2 * i] = get_point_f(def.point_tag_0, def.point_0);
        local.spts[2 * i + 1] = get_point_f(def.point_tag_1, def.point_1);
        auto &line = local.lines[i];
        line = line_t{};
        // a rounding moves a point by strictly less than one lattice
        // unit, so two coordinates two apart cannot be one point: the
        // exact test runs only inside that band
        bool banded = true;
        for (int d = 0; d < 3 && banded; ++d) {
          const auto delta =
              T1(local.spts[2 * i][d]) - T1(local.spts[2 * i + 1][d]);
          banded = delta <= T1(1) && delta >= T1(-1);
        }
        if (banded) {
          if (local.spts[2 * i][0] == local.spts[2 * i + 1][0] &&
              local.spts[2 * i][1] == local.spts[2 * i + 1][1] &&
              local.spts[2 * i][2] == local.spts[2 * i + 1][2])
            ++local.out.census.degenerate_materialized;
          if (plane_generators_equal<Index, Int>(g0, g1)) {
            local.cedges[i].valid = 0;
            ++local.out.census.degenerate_exact;
            // The proof is a WELD when the endpoints are two identities:
            // one position, two names. The identity gate is the one
            // producer of that merge and it absorbs every identity the
            // roots of a statement name, so the coincidence is stated
            // there — on the proof alone, in every plane that proves it.
            // One identity has nothing to state.
            const auto name_0 = endpoint_name(i, 0);
            const auto name_1 = endpoint_name(i, 1);
            if (name_0 != name_1) {
              const auto canon = local.cedges[i].canon;
              emit_landing(canon, canon, name_0, local.out);
              emit_landing(canon, canon, name_1, local.out);
            }
            continue;
          }
        }
        build_plane_line(g0, g1, flat_0, flat_1, def, is_member, face_points,
                         edge_frame, ax0, ax1, line);
        local.cedges[i].valid = line.valid ? 1 : 0;
      }

      // The original features a canonical edge contributes to a
      // junction's name: the ORIGINAL EDGE it lies on if it lies on
      // one, else the PLANES it is the intersection of — a seam lies
      // in two, its own and its partner's, and the set is a property
      // of the canonical edge, so every plane that observes it names
      // the same features.
      auto edge_feature = [&](Index canon, bool &is_carrier,
                              std::array<Index, 4> &planes, int &n_planes) {
        is_carrier = false;
        n_planes = 0;
        for (const auto &instance : g.canon_group(canon)) {
          if (instance.side < 0)
            continue;
          pt3_t ga, gb;
          Index fa = Index(-1), fb = Index(-1);
          if (edge_frame(instance, ga, gb, fa, fb)) {
            is_carrier = true;
            planes[0] = fa;
            planes[1] = fb;
            n_planes = 2;
            return;
          }
        }
        auto add = [&](Index p) {
          if (p == Index(-1))
            return;
          for (int k = 0; k < n_planes; ++k)
            if (planes[std::size_t(k)] == p)
              return;
          if (n_planes < 4)
            planes[std::size_t(n_planes++)] = p;
        };
        for (const auto &instance : g.canon_group(canon)) {
          add(g.plane_of()[std::size_t(instance.face)]);
          const auto face = group_of(instance.tag_other, instance.object_other);
          if (face != Index(-1))
            add(g.plane_of()[std::size_t(face)]);
        }
        std::sort(planes.begin(), planes.begin() + n_planes);
      };
      auto junction_name = [&](Index canon_a, Index canon_b,
                               wide_t &sink) -> plane_name<Index> {
        bool carrier_a = false, carrier_b = false;
        std::array<Index, 4> fa{}, fb{};
        int na = 0, nb = 0;
        edge_feature(canon_a, carrier_a, fa, na);
        edge_feature(canon_b, carrier_b, fb, nb);
        if (carrier_a && carrier_b) {
          ++sink.census.names_edge_edge;
          std::array<Index, 2> x{fa[0], fa[1]}, y{fb[0], fb[1]};
          if (y < x)
            std::swap(x, y);
          return {Index(4), x[0], x[1], y[0], y[1]};
        }
        if (carrier_a || carrier_b) {
          ++sink.census.names_edge_plane;
          const auto &carrier = carrier_a ? fa : fb;
          const auto &others = carrier_a ? fb : fa;
          const auto n_others = carrier_a ? nb : na;
          // the plane of the pair that is NOT the carrier's own
          Index cut = Index(-1);
          for (int k = 0; k < n_others; ++k)
            if (others[std::size_t(k)] != plane &&
                (cut == Index(-1) || others[std::size_t(k)] < cut))
              cut = others[std::size_t(k)];
          return {Index(3), carrier[0], carrier[1],
                  cut == Index(-1) ? plane : cut, Index(-1)};
        }
        // the union of the two lines' plane sets: they share the
        // carrier plane, so a proper crossing names three
        std::array<Index, 8> merged{};
        int n = 0;
        auto add = [&](Index p) {
          for (int k = 0; k < n; ++k)
            if (merged[std::size_t(k)] == p)
              return;
          merged[std::size_t(n++)] = p;
        };
        for (int k = 0; k < na; ++k)
          add(fa[std::size_t(k)]);
        for (int k = 0; k < nb; ++k)
          add(fb[std::size_t(k)]);
        std::sort(merged.begin(), merged.begin() + n);
        if (n > 3)
          ++sink.census.names_truncated;
        ++sink.census.names_triple;
        return {Index(2), merged[0], n > 1 ? merged[1] : Index(-1),
                n > 2 ? merged[2] : Index(-1), Index(-1)};
      };

      auto test_pair = [&](std::size_t i, std::size_t j,
                           wide_t &sink) {
        const auto &ca = local.cedges[i];
        const auto &cb = local.cedges[j];
        if (!ca.valid || !cb.valid)
          return;
        ++sink.census.pairs_tested;
        const auto &la = local.lines[i];
        const auto &lb = local.lines[j];
        const auto &a0 = local.gens[2 * i];
        const auto &a1 = local.gens[2 * i + 1];
        const auto &b0 = local.gens[2 * j];
        const auto &b1 = local.gens[2 * j + 1];
        const int sb0 = plane_line_sign<Index, Int>(la, b0, ax0, ax1);
        const int sb1 = plane_line_sign<Index, Int>(la, b1, ax0, ax1);
        if (sb0 * sb1 > 0)
          return;
        const int sa0 = plane_line_sign<Index, Int>(lb, a0, ax0, ax1);
        const int sa1 = plane_line_sign<Index, Int>(lb, a1, ax0, ax1);
        if (sa0 * sa1 > 0)
          return;

        // alignment is an identity fact, not a crossing to resolve:
        // aligned pairs resolve no matter what the contour flags say
        if (sa0 == 0 && sa1 == 0 && sb0 == 0 && sb1 == 0) {
          ++sink.census.collinear_pairs;
          plane_collinear_pair(local, sink, i, j, la, lb, own_a, own_b, own_c,
                               emit_landing, endpoint_name);
          return;
        }
        if (!resolve_self_contours && ca.contour != std::int16_t(-2) &&
            cb.contour != std::int16_t(-2) && ca.contour == cb.contour)
          return;

        // an endpoint on the other line: the junction is that
        // endpoint's own identity, and the other edge's endpoints
        // straddling THIS line is exactly "strictly inside its span"
        bool incident = false;
        if (sa0 == 0) {
          incident = true;
          if (sb0 * sb1 < 0)
            emit_landing(cb.canon, ca.canon, endpoint_name(i, 0), sink);
        }
        if (sa1 == 0) {
          incident = true;
          if (sb0 * sb1 < 0)
            emit_landing(cb.canon, ca.canon, endpoint_name(i, 1), sink);
        }
        if (sb0 == 0) {
          incident = true;
          if (sa0 * sa1 < 0)
            emit_landing(ca.canon, cb.canon, endpoint_name(j, 0), sink);
        }
        if (sb1 == 0) {
          incident = true;
          if (sa0 * sa1 < 0)
            emit_landing(ca.canon, cb.canon, endpoint_name(j, 1), sink);
        }
        if (incident)
          return;

        // a proper crossing: one name, stated on both roots
        const auto name = junction_name(ca.canon, cb.canon, sink);
        sink.statements.push_back({name, ca.canon, cb.canon, plane});
        sink.statements.push_back({name, cb.canon, ca.canon, plane});
        ++sink.census.crossings_ee;
      };

      if (n <= 16) {
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = i + 1; j < n; ++j)
            test_pair(i, j, local.out);
      } else {
        local.pts_2d.clear();
        local.seg_ids.clear();
        local.pts_2d.reserve(n * 2);
        local.seg_ids.reserve(n * 2);
        for (std::size_t i = 0; i < n; ++i) {
          local.pts_2d.emplace_back(local.spts[2 * i][ax0],
                                    local.spts[2 * i][ax1]);
          local.pts_2d.emplace_back(local.spts[2 * i + 1][ax0],
                                    local.spts[2 * i + 1][ax1]);
          local.seg_ids.push_back(int(2 * i));
          local.seg_ids.push_back(int(2 * i + 1));
        }
        auto segments = tf::make_segments(local.seg_ids, local.pts_2d);
        local.tree.build(segments, tf::config_tree(4, 4));
        if (n < wide_carrier_edges) {
          tf::search_self(
              local.tree, tf::intersects_f,
              [&](int id0, int id1) {
                test_pair(std::size_t(id0 < id1 ? id0 : id1),
                          std::size_t(id0 < id1 ? id1 : id0), local.out);
              },
              /*parallelism_depth=*/0);
        } else {
          // THE GRAIN MOVES INSIDE THE CARRIER. A pass costs its own work,
          // never its largest carrier: one plane can hold half a pair's
          // edges, and dealing carriers between chunks cannot divide a
          // carrier that is itself the tail. So the wide carrier is walked
          // at the EDGE, each edge asking the same tree the self-search
          // asks and stating the same pairs — one kernel, one verdict, a
          // second grain.
          //
          // The statements are a SET, not a sequence: their consumer sorts
          // them on the whole record and elects each class's first, so the
          // order they arrive in is a schedule and not a product. The
          // aggregation is sequenced all the same, so the buffer this
          // builds is the same buffer at every thread count.
          tf::blocked_reduce_sequenced_aggregate(
              tf::make_sequence_range(Index(n)), tf::none, wide_t{},
              [&](auto &&sub, wide_t &wide) {
                for (const auto i : sub) {
                  const auto q0 = local.pts_2d[std::size_t(2 * i)];
                  const auto q1 = local.pts_2d[std::size_t(2 * i + 1)];
                  const auto query = tf::make_aabb(
                      tf::point<Int, 2>{q0[0] < q1[0] ? q0[0] : q1[0],
                                        q0[1] < q1[1] ? q0[1] : q1[1]},
                      tf::point<Int, 2>{q0[0] > q1[0] ? q0[0] : q1[0],
                                        q0[1] > q1[1] ? q0[1] : q1[1]});
                  tf::search(
                      local.tree,
                      [&](const auto &bv) { return tf::intersects(bv, query); },
                      [&](int id) {
                        if (id > int(i))
                          test_pair(std::size_t(i), std::size_t(id), wide);
                      });
                }
              },
              [&](const wide_t &wide, const tf::none_t &) {
                tf::core::append(wide.statements, local.out.statements);
                local.out.census += wide.census;
              });
        }
      }
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.out.statements, statements);
    census += local.out.census;
  };
  tf::blocked_reduce_sequenced_aggregate(tf::enumerate(g.plane_faces()),
                                         tf::none, local_t{}, task, agg);
}

} // namespace tf::intersect::graph
