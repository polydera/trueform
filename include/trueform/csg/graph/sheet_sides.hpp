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
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The side every domain takes of every sheet, one state per
///        `(sheet, domain)`, sheet-major.
///
/// A sheet has no outside, so no side holds by definition: a side is a
/// statement, and a domain the sheet does not separate carries none.
///
/// The local side. A wall that carries a layer of the sheet of odd count
/// states both of its domains: the one on the side the count enters lies
/// behind the sheet, the other in front. A domain on both sides of one such
/// wall, or stated both ways by two of them, wraps the sheet.
///
/// The flush witness. A sheet passing through a fan's edge stands on two of
/// its pages; one ending there stands on one, and the other half-plane of
/// that page's carrier lies inside a sector. That sector is the two sectors
/// a passing sheet would divide, collapsed into one domain that stands on
/// both sides of the missing page — the same domain twice around the fan —
/// so it wraps the sheet, whatever side a wall states for it elsewhere.
///
/// The nested witness. A free rim — an open component of the sheet — lies
/// inside the region its bundle lands in, which reaches both of its faces
/// around it and wraps the sheet. A nesting merge unites two false halves of
/// one region, so a wrap of either is a wrap of both.
///
/// No side, no transit. A domain that wraps the sheet has no side, and no
/// side is carried through it. Every other side is copied across the walls
/// that do not carry the sheet; a wall that does carry it has both of its
/// domains stated already. A domain no statement reaches has no side
/// either, and a domain without a side reads zero. A bundle whose envelope
/// wraps the sheet lies wholly on one side of it all the same: its winding
/// seed states the bundle's own domains instead of the envelope.
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

  /// Whether component `c` carries a layer of sheet `k` of odd count.
  auto carries(const arrangement_descriptor<Index> &desc,
               const component_crossings<Index> &crossings, Index c,
               Index k) const -> bool {
    bool found = false;
    for_each_sheet_layer(desc, crossings, c,
                         [&](Index s, Index) { found = found || s == k; });
    return found;
  }
};

} // namespace tf::csg::graph
