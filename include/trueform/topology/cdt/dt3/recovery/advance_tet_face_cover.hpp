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
#include "../../../../core/buffer.hpp"
#include "../../../../core/config/verify.hpp"
#include "../find_tet_edge.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_state.hpp"
#include "./spread_tet_face_cover.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_rim_piece.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Advance a face's cover, read by @ref read_tet_face_cover and held, across
/// the rows `owner` wrote since its `_written` held `mark` rows. A held facet
/// leaves the complex only where a point placed on it divides it, into
/// facets its face holds as it held the one
/// (@ref split_tetrahedralization_site), so the cover only grows and its
/// `frontier` changes in two ways alone. A facet stands across a piece that
/// an edge of a live row written since names, since an edge no such row
/// holds kept its star. And the piece `divided` names — the edge the append
/// of the point placed since divided, slots ascending, then the point's
/// site; `none` thrice where it divided none — stands no more, its halves
/// about the site standing for it on its side. Those pieces and halves are
/// the seeds; the
/// cover spreads from them alone (@ref spread_tet_face_cover), `tickets`
/// receiving the facets it gained, and `frontier` becomes the pieces kept
/// with those the spread left, sorted. `touched`, `seeds` and `hashed` are
/// scratch.
/// Every site is read in the decision's `sites` on the projection `axes`.
template <typename Owner, typename Sites, typename Index>
auto advance_tet_face_cover(const Owner &owner, const Sites &sites,
                            const std::array<Index, 3> &plane,
                            const std::pair<int, int> &axes, std::size_t mark,
                            const std::array<Index, 3> &divided,
                            tf::buffer<std::array<Index, 2>> &touched,
                            tf::buffer<tet_rim_piece<Index>> &seeds,
                            tf::buffer<Index> &tickets,
                            tf::buffer<tet_rim_piece<Index>> &frontier,
                            tet_face_cover_scratch<Index> &scratch) -> void {
  tickets.clear();
  touched.clear();
  seeds.clear();
  for (std::size_t i = mark; i < owner._written.size(); ++i) {
    const Index row = owner._written[i].row;
    if (owner._states[std::size_t(row)] != tet_state::live)
      continue;
    const auto corners = owner._corners[std::size_t(row)];
    for (std::size_t a = 0; a < 4; ++a)
      for (std::size_t b = a + 1; b < 4; ++b)
        touched.push_back({std::min(corners[a], corners[b]),
                           std::max(corners[a], corners[b])});
  }
  std::sort(touched.begin(), touched.end());
  touched.erase_till_end(std::unique(touched.begin(), touched.end()));
  std::size_t kept = 0, t = 0;
  bool halved = false;
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    const auto piece = frontier[i];
    const std::array<Index, 2> key{piece.lo, piece.hi};
    if (key[0] == divided[0] && key[1] == divided[1]) {
      const Index site = divided[2];
      if constexpr (tf::core::verification) {
        tet_edge_ref<Index> standing{};
        tf::core::verify(
            !find_tet_edge(owner, piece.lo, piece.hi, standing, scratch.star),
            "the piece an advance divides the complex holds no "
            "more");
      }
      seeds.push_back(piece.lo < site
                          ? tet_rim_piece<Index>{piece.lo, site, piece.side}
                          : tet_rim_piece<Index>{site, piece.lo, -piece.side});
      seeds.push_back(site < piece.hi
                          ? tet_rim_piece<Index>{site, piece.hi, piece.side}
                          : tet_rim_piece<Index>{piece.hi, site, -piece.side});
      halved = true;
      continue;
    }
    while (t < touched.size() && touched[t] < key)
      ++t;
    if (t < touched.size() && touched[t] == key)
      seeds.push_back(piece);
    else
      frontier[kept++] = piece;
  }
  if (!seeds.size())
    return;
  if (halved)
    std::sort(seeds.begin(), seeds.end(), tet_rim_piece_precedes<Index>);
  frontier.erase_till_end(frontier.begin() + std::ptrdiff_t(kept));
  spread_tet_face_cover(owner, sites, plane, axes, seeds, tickets, frontier,
                        scratch);
  const auto grown = frontier.begin() + std::ptrdiff_t(kept);
  std::sort(grown, frontier.end(), tet_rim_piece_precedes<Index>);
  seeds.allocate(frontier.size());
  std::merge(frontier.begin(), grown, grown, frontier.end(), seeds.begin(),
             tet_rim_piece_precedes<Index>);
  std::swap(seeds, frontier);
}

} // namespace tf::topology::cdt::dt3
