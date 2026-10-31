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
#include "../../core/buffer.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The side every domain takes of every sheet, one state per
///        `(sheet, domain)`, sheet-major.
///
/// A sheet has no outside, so a side is a statement, and no side is ever
/// carried from one domain to another.
///
/// Inside a volume a sheet separates or it does not, and what it separates
/// is a component of the volume's inside: its domains joined across every
/// wall the sheet is not on. A component the sheet's walls reach from one
/// side only lies on that side. One they reach from both sides goes around
/// a rim inside the volume: the sheet does not separate it, and its domains
/// have no side. One they never reach states nothing. A domain inside
/// several volumes takes a one-side statement over a wrap, and two one-side
/// statements that disagree prove the union is not separated: no side, and
/// the disagreement is counted.
///
/// A free rim — an open component carrying a layer of the sheet — joins the
/// two domains it stands between, so a volume's inside reached around the
/// rim is one component that sees the sheet from both sides; the join only
/// ever unites components. Both of the rim's sides inside a volume mean the
/// rim is interior and the pair is counted; one side out means the rim
/// stands on the volume's own wall and the sheet runs edge to edge. Nesting
/// never silences a bundle's own separation: a volume the sheet separates
/// states its side wherever it sits.
///
/// The flush witness. A sheet passing through a fan's edge stands on two of
/// its pages; one ending there stands on one, and the other half-plane of
/// that page's carrier lies inside a sector. That sector is the two sectors
/// a passing sheet would divide, collapsed into one domain that stands on
/// both sides of the missing page, so it wraps the sheet — unless a volume
/// states it a side.
///
/// Every other domain takes the winding's side at a point of one of its
/// walls the sheet is not on — the side of the sheet's own extension through
/// the winding field's zero set. A domain every wall of which the sheet is
/// on has no such point; the sheet's walls state it as they do a component.
/// A domain without a side reads zero.
template <typename Index> struct sheet_sides {
  static constexpr char unstated = 0;
  static constexpr char front = 1;
  static constexpr char behind = 2;
  static constexpr char wraps = 3;

  /// Per sheet: its form tag, ascending.
  tf::buffer<Index> tags;
  /// Per form tag: its sheet, `-1` for a volume.
  tf::buffer<Index> sheet_of_tag;
  Index n_domains = 0;
  /// `states[k * n_domains + d]`: the side domain `d` takes of sheet `k`.
  tf::buffer<char> states;
  /// The `(sheet tag, volume tag)` pairs whose volume holds a free rim of
  /// the sheet, ascending.
  tf::buffer<std::array<Index, 2>> rim_volumes;
  /// The `(sheet, domain)` pairs two volumes stated opposite sides of.
  Index n_conflicts = 0;

  auto n_sheets() const -> Index { return Index(tags.size()); }

  auto state(Index k, Index d) const -> char {
    return states[static_cast<std::size_t>(k) *
                      static_cast<std::size_t>(n_domains) +
                  static_cast<std::size_t>(d)];
  }
  auto state(Index k, Index d) -> char & {
    return states[static_cast<std::size_t>(k) *
                      static_cast<std::size_t>(n_domains) +
                  static_cast<std::size_t>(d)];
  }

  /// Calls `f(k, behind_side)` for every sheet `k` component `c` carries a
  /// layer of odd count of, `behind_side` the side of `c` lying behind it.
  template <typename F>
  auto for_each_sheet_layer(const arrangement_descriptor<Index> &desc,
                            const component_crossings<Index> &crossings,
                            Index c, const F &f) const -> void {
    const Index row = crossings.row_of(c);
    if (row == Index(-1)) {
      const Index t = desc.tag_of_component[c];
      if (t >= Index(0) && sheet_of_tag[t] >= Index(0))
        f(sheet_of_tag[t], Index(1));
      return;
    }
    for (const auto &layer : crossings.layers[static_cast<std::size_t>(row)]) {
      const Index k = sheet_of_tag[layer[0]];
      if (k >= Index(0) && layer[1] % Index(2) != Index(0))
        f(k, layer[1] > Index(0) ? Index(1) : Index(0));
    }
  }

  /// The sides of sheet `k` its walls state domain `d` on: `front`,
  /// `behind`, both (`wraps`) or none (`unstated`).
  template <typename DomainComponents>
  auto walled(const arrangement_descriptor<Index> &desc,
              const component_crossings<Index> &crossings,
              const DomainComponents &domain_components, Index k,
              Index d) const -> char {
    char s = unstated;
    for (const Index c : domain_components[std::size_t(d)])
      for_each_sheet_layer(desc, crossings, c, [&](Index j, Index behind_side) {
        if (j != k)
          return;
        if (desc.domain_of_side[2 * c + behind_side] == d)
          s = char(s | behind);
        if (desc.domain_of_side[2 * c + 1 - behind_side] == d)
          s = char(s | front);
      });
    return s;
  }

  /// Whether component `c` carries a layer of sheet `k` of odd count.
  auto carries(const arrangement_descriptor<Index> &desc,
               const component_crossings<Index> &crossings, Index c,
               Index k) const -> bool {
    bool found = false;
    for_each_sheet_layer(desc, crossings, c,
                         [&](Index s, Index) { found = found || s == k; });
    return found;
  }

  /// Whether sheet `k` lies on component `c` at all, a cancelled layer
  /// included: no point of `c` is then off the sheet.
  auto touches(const arrangement_descriptor<Index> &desc,
               const component_crossings<Index> &crossings, Index c,
               Index k) const -> bool {
    const Index row = crossings.row_of(c);
    if (row == Index(-1)) {
      const Index t = desc.tag_of_component[c];
      return t >= Index(0) && sheet_of_tag[t] == k;
    }
    for (const auto &layer : crossings.layers[static_cast<std::size_t>(row)])
      if (sheet_of_tag[layer[0]] == k)
        return true;
    return false;
  }
};

} // namespace tf::csg::graph
