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

#include "../../exact/edge_edge_parameter.hpp"
#include "../../exact/edge_parameter.hpp"
#include "../../exact/edge_plane_parameter.hpp"
#include "../../exact/edge_point_parameter.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex.hpp"

#include <array>
#include <cstddef>

namespace tf::exact {

/// What a face-pair workspace stores per emitted record: the exact
/// parameter of the point on every original edge it lies on.
///
/// The two slots are ordered by carrier — the unordered pair of the
/// edge's flat vertex ids — and each parameter is measured from its
/// carrier's lower flat id. Both conventions are functions of original
/// vertex ids alone, which is what makes the payload survive
/// duplication: a copy is rewritten onto another face, has its two
/// sides transposed, and runs its edge the other way round, and none of
/// that touches a vertex id. A record with one edge fills slot 0; a
/// record with none fills nothing and exists only so that a record id
/// remains a dense count of emissions.
template <typename Int, typename Index> struct edge_fractions {
  std::array<edge_parameter<Int>, 2> t;
  /// The edges the fractions live on, as flat vertex ids — and for a
  /// vertex-involving fact, the vertex itself: slot 0 = the vertex's
  /// flat id twice (VV: the two identified vertices). A slot with no
  /// content holds the id pair {-1, -1}.
  std::array<Index, 2> from{Index(-1), Index(-1)};
  std::array<Index, 2> to{Index(-1), Index(-1)};
};

/// Ascending order of two carriers named by their endpoints' flat vertex
/// ids. One producer for both the kernel that writes the two fractions
/// and the stage that picks one of them.
template <typename Index>
auto carrier_key_less(Index a0, Index a1, Index b0, Index b1) -> bool {
  const auto au = a0 < a1 ? a0 : a1, av = a0 < a1 ? a1 : a0;
  const auto bu = b0 < b1 ? b0 : b1, bv = b0 < b1 ? b1 : b0;
  return au != bu ? au < bu : av < bv;
}

/// A parameter on the edge (`from`, `to`), restated in the direction the
/// carrier is stored in: from its lower flat vertex id.
template <typename Int, typename Index>
auto carrier_oriented_parameter(const edge_parameter<Int> &t, Index from,
                                Index to) -> edge_parameter<Int> {
  return from < to ? t : reversed_parameter(t);
}

/// A point on no carrier: a shared vertex, or a pierce through a face's
/// interior.
template <typename Int, typename Index>
auto make_point_payload(Index va, Index vb) -> edge_fractions<Int, Index> {
  edge_fractions<Int, Index> out{};
  out.from[0] = va;
  out.to[0] = vb;
  return out;
}

/// An edge crossing a face's supporting plane. `values[i0]`, `values[i1]`
/// are that plane's orient3d values at the edge's endpoints — the
/// classifier's own output, so the fraction is not a second derivation
/// of a fact the sign mask already proved.
template <typename Index, typename Int, typename Values>
auto make_edge_plane_payload(const vertex<Index, Int> &v0,
                             const vertex<Index, Int> &v1, const Values &values,
                             std::size_t i0, std::size_t i1)
    -> edge_fractions<Int, Index> {
  using T2 = typename meta<Int>::T2;
  auto num = values[i0];
  auto den = values[i0] - values[i1];
  if (den < T2(0)) {
    num = -num;
    den = -den;
  }
  edge_fractions<Int, Index> out{};
  out.t[0] = carrier_oriented_parameter<Int>({num, den}, v0.id, v1.id);
  out.from[0] = v0.id;
  out.to[0] = v1.id;
  return out;
}

/// The SoS flavour of the same fact: the fraction is the exact crossing
/// of (v0, v1) with the plane of (a, b, c) — the face's own supporting
/// plane, which is what the identity tier names, not the fan triangle the
/// perturbation happened to hit.
template <typename Index, typename Int>
auto make_sos_edge_plane_payload(const pt3<Int> &a, const pt3<Int> &b,
                                 const pt3<Int> &c,
                                 const vertex<Index, Int> &v0,
                                 const vertex<Index, Int> &v1)
    -> edge_fractions<Int, Index> {
  edge_fractions<Int, Index> out{};
  out.t[0] = carrier_oriented_parameter<Int>(
      make_edge_plane_parameter<Int>(a, b, c, v0.pt, v1.pt), v0.id, v1.id);
  out.from[0] = v0.id;
  out.to[0] = v1.id;
  return out;
}

/// A vertex `q` lying on the edge (v0, v1). The incidence is the
/// caller's statement; the fraction is then exactly q's position.
template <typename Index, typename Int>
auto make_vertex_edge_payload(const vertex<Index, Int> &v0,
                              const vertex<Index, Int> &v1, const pt3<Int> &q,
                              Index q_id) -> edge_fractions<Int, Index> {
  edge_fractions<Int, Index> out{};
  out.t[0] = carrier_oriented_parameter<Int>(
      make_edge_point_parameter<Int>(v0.pt, v1.pt, q), v0.id, v1.id);
  out.from[0] = v0.id;
  out.to[0] = v1.id;
  out.from[1] = q_id;
  out.to[1] = q_id;
  return out;
}

/// Two edges meeting. Both fractions describe one point, so the
/// projection the crossing is solved in is shared; reversing either edge
/// leaves the resulting parameter pair unchanged, which is why the
/// axes may be taken once.
template <typename Index, typename Int>
auto make_edge_edge_payload(const vertex<Index, Int> &a0,
                            const vertex<Index, Int> &a1,
                            const vertex<Index, Int> &b0,
                            const vertex<Index, Int> &b1)
    -> edge_fractions<Int, Index> {
  const auto axes = projection_axes_edges<Int>(a0.pt, a1.pt, b0.pt, b1.pt);
  const auto ta = carrier_oriented_parameter<Int>(
      make_edge_edge_parameter<Int>(a0.pt, a1.pt, b0.pt, b1.pt, axes), a0.id,
      a1.id);
  const auto tb = carrier_oriented_parameter<Int>(
      make_edge_edge_parameter<Int>(b0.pt, b1.pt, a0.pt, a1.pt, axes), b0.id,
      b1.id);
  edge_fractions<Int, Index> out{};
  const bool a_first = carrier_key_less(a0.id, a1.id, b0.id, b1.id);
  out.t[0] = a_first ? ta : tb;
  out.t[1] = a_first ? tb : ta;
  out.from[0] = a_first ? a0.id : b0.id;
  out.to[0] = a_first ? a1.id : b1.id;
  out.from[1] = a_first ? b0.id : a0.id;
  out.to[1] = a_first ? b1.id : a1.id;
  return out;
}

} // namespace tf::exact
