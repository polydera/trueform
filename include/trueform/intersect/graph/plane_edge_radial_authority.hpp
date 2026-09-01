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

#include "../../exact/det2_sign.hpp"
#include "../../exact/meta.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_pair_carrier.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// THE RADIAL AUTHORITY of one canonical piece: the carrier line every
/// definition of that piece stands on, oriented along the piece's OWN key
/// order.
///
/// The line is published as the axis its largest component falls on and that
/// component's sign, because that is the whole of what a radial ring reads —
/// the turn of two half-planes around the line is the sign of their cross on
/// that axis — and because it is the only form the width ladder holds for
/// every class: an intersection edge's line is `n_f x n_g`, degree four, which
/// has no rung, while each of its component SIGNS is one
/// @ref tf::exact::det2_sign away.
///
/// `valid` false is a stated REFUSAL: the piece's definitions do not name one
/// line, so no order around it is meaningful and the consumer is told so
/// rather than handed a number.
struct plane_edge_radial_authority {
  int axis = 0;
  int sign = 0;
  bool valid = false;
};

/// THE ONE PRODUCER of a piece's radial line, read off the complete
/// definition span of that piece — the only place every statement about it is
/// in hand at once.
///
/// A definition names the line it stands on in one of two languages, and the
/// class of the piece is which languages are present:
///
/// - AN ORIGINAL EDGE (`side >= 0`) states it as tier-1 input data: the two
///   ORIGINAL vertices of `(tag_other, object_other, side)`, traversed in the
///   face's own winding, with
///   @ref tf::intersect::graph::plane_edge_reversed_flag saying whether key
///   order runs the other way. Splitting never changes that line — a piece
///   inherits its parent's side and carries the flip through
///   @ref tf::intersect::graph::plane_piece_def — so a split piece's line is
///   its parent's, exactly.
/// - AN INTERIOR CUT (@ref tf::intersect::graph::plane_edge_radial_flag)
///   states it as its producing pair's carrier `n_own x n_other`, with the
///   flag pair saying which way key order runs along it. The pair is the
///   definition's own face and the partner it carries, so the statement is
///   reproduced from generators here and never from a materialized position.
///
/// THE CLASS PRECEDENCE. Where both are present the ORIGINAL EDGE is the
/// canonical line: it is input data of the operand mesh, exact at its own
/// width, and it is the same line at every piece the edge was split into,
/// while a pair carrier is a derived degree-four cross whose orientation
/// depends on which of the two faces states it. The cut statements are then a
/// free cross-check of it, never a competing answer.
///
/// THE RECONCILIATION. A piece may carry many statements — one per face
/// meeting there, each against ITS pair — so agreement is not one bit. Every
/// other statement must name the SAME line as the representative (exactly:
/// zero cross for two original edges, and for a pair carrier, that the
/// representative's line lies in both of the pair's planes) and, re-expressed
/// against the representative's direction, must give the SAME key-order
/// sense. Anything else is ill-posed — distinct edges welded into one
/// canonical identity — and refuses.
template <typename Index, typename Int, typename Definitions,
          typename DescriptorOfFace, typename ApplyToFace, typename GetPoint>
auto make_plane_edge_radial_authority(
    const Definitions &definitions, const DescriptorOfFace &descriptor_of_face,
    const ApplyToFace &apply_to_face, const GetPoint &get_point)
    -> plane_edge_radial_authority {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  plane_edge_radial_authority authority;

  const auto magnitude = [](const T1 &value) {
    return value < T1(0) ? T1(-value) : value;
  };
  // the original edge a boundary definition lies on, turned into the
  // definition's own key order
  const auto original_edge = [&](const plane_edge_def<Index> &def,
                                 std::array<T1, 3> &line) {
    bool stated = false;
    apply_to_face(int(def.tag_other), def.object_other,
                  [&](const auto &corners) {
                    const auto n = corners.size();
                    if (std::size_t(def.side) >= n)
                      return;
                    const auto from = get_point(
                        int(def.tag_other),
                        Index(corners[std::size_t(def.side)]));
                    const auto to = get_point(
                        int(def.tag_other),
                        Index(corners[(std::size_t(def.side) + 1) % n]));
                    line = {T1(to[0]) - from[0], T1(to[1]) - from[1],
                            T1(to[2]) - from[2]};
                    stated = line[0] != T1(0) || line[1] != T1(0) ||
                             line[2] != T1(0);
                  });
    if (stated && (def.flags & plane_edge_reversed_flag) != 0)
      for (int c = 0; c < 3; ++c)
        line[std::size_t(c)] = -line[std::size_t(c)];
    return stated;
  };
  // the producing pair an interior cut names, with both plane normals: the
  // definition's own face and the partner it carries
  const auto pair_carrier = [&](const plane_edge_def<Index> &def,
                                plane_pair_carrier<Int> &carrier,
                                std::array<T2, 3> &own,
                                std::array<T2, 3> &other) {
    const auto &descriptor = descriptor_of_face(def.face);
    std::array<T1, 3> a1{}, a2{}, b1{}, b2{};
    bool stated = false;
    apply_to_face(int(descriptor.tag), descriptor.object,
                  [&](const auto &corners) {
                    stated = plane_face_basis<Int>(
                        corners,
                        [&](std::size_t k) {
                          return get_point(int(descriptor.tag),
                                           Index(corners[k]));
                        },
                        a1, a2);
                  });
    if (!stated)
      return false;
    stated = false;
    apply_to_face(int(def.tag_other), def.object_other,
                  [&](const auto &corners) {
                    stated = plane_face_basis<Int>(
                        corners,
                        [&](std::size_t k) {
                          return get_point(int(def.tag_other),
                                           Index(corners[k]));
                        },
                        b1, b2);
                  });
    if (!stated)
      return false;
    carrier = make_plane_pair_carrier<Int>(a1, a2, b1, b2);
    own = plane_carrier_cross<Int>(a1, a2);
    other = plane_carrier_cross<Int>(b1, b2);
    return true;
  };
  // the carrier line is `a * b1 - b * b2`, so one component's sign is one
  // determinant and no degree-four value is ever formed
  const auto component_sign = [](const plane_pair_carrier<Int> &carrier,
                                 int axis) {
    return tf::exact::det2_sign<Int>(
        carrier.a, T2(carrier.b1[std::size_t(axis)]), carrier.b,
        T2(carrier.b2[std::size_t(axis)]));
  };
  const auto wider_component = [](const plane_pair_carrier<Int> &carrier,
                                  int axis, int sign, int against,
                                  int against_sign) {
    const auto turn = [](const T1 &value, int by) {
      return by > 0 ? value : T1(-value);
    };
    return tf::exact::det2_sign<Int>(
               carrier.a,
               T2(turn(carrier.b1[std::size_t(axis)], sign) -
                  turn(carrier.b1[std::size_t(against)], against_sign)),
               carrier.b,
               T2(turn(carrier.b2[std::size_t(axis)], sign) -
                  turn(carrier.b2[std::size_t(against)], against_sign))) > 0;
  };

  const auto n = std::size_t(definitions.size());
  auto boundary = n;
  auto cut = n;
  for (std::size_t k = 0; k < n; ++k) {
    const auto &def = definitions[k];
    if (def.side >= std::int16_t(0)) {
      if (boundary == n)
        boundary = k;
    } else if ((def.flags & plane_edge_radial_flag) != 0 && cut == n)
      cut = k;
  }

  if (boundary != n) {
    std::array<T1, 3> line{};
    if (!original_edge(definitions[boundary], line))
      return authority;
    int axis = 0;
    for (int c = 1; c < 3; ++c)
      if (magnitude(line[std::size_t(c)]) > magnitude(line[std::size_t(axis)]))
        axis = c;
    for (std::size_t k = 0; k < n; ++k) {
      if (k == boundary)
        continue;
      const auto &def = definitions[k];
      if (def.side >= std::int16_t(0)) {
        std::array<T1, 3> stated{};
        if (!original_edge(def, stated))
          return authority;
        // one line, one direction: the cross vanishes and the dot is positive
        if (T2(line[1]) * stated[2] - T2(line[2]) * stated[1] != T2(0) ||
            T2(line[2]) * stated[0] - T2(line[0]) * stated[2] != T2(0) ||
            T2(line[0]) * stated[1] - T2(line[1]) * stated[0] != T2(0))
          return authority;
        if (T2(line[0]) * stated[0] + T2(line[1]) * stated[1] +
                T2(line[2]) * stated[2] <=
            T2(0))
          return authority;
        continue;
      }
      if ((def.flags & plane_edge_radial_flag) == 0)
        continue;
      plane_pair_carrier<Int> carrier;
      std::array<T2, 3> own{}, other{};
      if (!pair_carrier(def, carrier, own, other))
        return authority;
      // the pair's own statement, read against the canonical direction: a
      // zero here is a line the pair does not carry at all
      if (plane_carrier_dot_sign<Int>(carrier, line) !=
          plane_edge_radial_sign(def))
        return authority;
    }
    authority.axis = axis;
    authority.sign = line[std::size_t(axis)] > T1(0) ? 1 : -1;
    authority.valid = true;
    return authority;
  }

  if (cut == n)
    return authority;
  plane_pair_carrier<Int> carrier;
  std::array<T2, 3> own{}, other{};
  if (!pair_carrier(definitions[cut], carrier, own, other))
    return authority;
  std::array<int, 3> component{};
  for (int c = 0; c < 3; ++c)
    component[std::size_t(c)] = component_sign(carrier, c);
  int axis = -1;
  for (int c = 0; c < 3; ++c) {
    if (component[std::size_t(c)] == 0)
      continue;
    if (axis < 0 ||
        wider_component(carrier, c, component[std::size_t(c)], axis,
                        component[std::size_t(axis)]))
      axis = c;
  }
  if (axis < 0)
    return authority;
  const auto stated = plane_edge_radial_sign(definitions[cut]);
  for (std::size_t k = 0; k < n; ++k) {
    if (k == cut)
      continue;
    const auto &def = definitions[k];
    if ((def.flags & plane_edge_radial_flag) == 0)
      continue;
    plane_pair_carrier<Int> partner;
    std::array<T2, 3> partner_own{}, partner_other{};
    if (!pair_carrier(def, partner, partner_own, partner_other))
      return authority;
    // one line: the representative's line lies in both of this pair's planes,
    // so it is parallel to their cross
    if (plane_carrier_dot_sign<Int>(carrier, partner_own) != 0 ||
        plane_carrier_dot_sign<Int>(carrier, partner_other) != 0)
      return authority;
    const auto relative =
        component_sign(partner, axis) * component[std::size_t(axis)];
    if (relative == 0 || relative * plane_edge_radial_sign(def) != stated)
      return authority;
  }
  authority.axis = axis;
  authority.sign = component[std::size_t(axis)] * stated;
  authority.valid = true;
  return authority;
}

} // namespace tf::intersect::graph
