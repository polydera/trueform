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
#include "../collapse_tet_site.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./ticket_tet_subsegment.hpp"
#include "./withdraw_tet_subsegment.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Collapse `site` into `survivor` (@ref collapse_tet_site) and carry the
/// segment rows naming it. The precondition is the census's barrier: the
/// segment phase stands at its fixed point, so every row naming the site is
/// a recovered edge of the complex, read among the rows ticketed at the
/// lower of the site and a site of its link; a row of a segment not
/// recovered is a refusal of its own, and a site whose star names one does
/// not collapse. A segment through the site and the survivor loses the site
/// from its chain, its two rows there joined into one; every other segment
/// through the site runs to the survivor instead, its numerators kept, each
/// row re-ticketed at its lower site. A segment ending at the site is the
/// surface's: every face along it holds the site as a corner, so the caller
/// retires the faces holding the site and states them on the survivor, as a
/// cut states its children (@ref retire_tet_faces, @ref retire_tet_segments,
/// @ref recover_tet_segment_phase), which retires every segment ending at the
/// site and states or finds the survivor's — a face holding both sites
/// states nothing, and a segment between them retires with its last face.
/// Its rows stand as they are until then; a segment no face bounds ending at
/// the site has no such statement, and the caller states none (the mesher
/// states rims alone). `_collapse.faces` names every face whose cover moved:
/// those holding the site as a corner are restated, the rest are the face
/// phase's due list. Each pass costs the site's star and the rows ticketed
/// at its link.
///
/// The census convicts by shape, and the collapse is an edge collapse: one
/// site onto a site its star holds, the survivor standing where it stood.
/// A needle — an input triangle two of whose sides stand within the input
/// quantum of each other at a stated point — collapses the short side's two
/// corners into one; a cap — a corner in a reach cell the opposite side
/// crosses —
/// collapses the corner onto the nearer corner of that side. Between two
/// original sites the election's smallest key survives. Either refuses where
/// the two are not an edge of the complex (a recovered segment between them
/// is one), where the site's star reaches the hull, where a rewritten cell
/// would not stand positive, and where the star's held facets cannot close
/// (@ref collapse_tet_site); a refused conviction stands as a refusal. A
/// survivor sharing no edge with the site would be a collapse along a path,
/// which this is not.
template <typename Owner, typename Coord>
auto collapse_tet_site_rows(
    Owner &owner,
    tet_segment_recovery<typename Owner::index_type, Coord> &recovery,
    typename Owner::index_type site, typename Owner::index_type survivor)
    -> bool {
  using Index = typename Owner::index_type;
  auto &link = recovery.collapse_link;
  auto &found = recovery.collapse_rows;
  link.clear();
  found.clear();
  auto &rows = recovery.subsegments;
  const auto names_site = [site](const tet_subsegment<Index> &row) {
    return row.a == site || row.b == site;
  };
  const auto consents = [&](const tf::buffer<Index> &star) {
    for (const auto cell : star)
      for (const auto corner : owner._corners[std::size_t(cell)])
        if (corner != site && corner != Owner::infinite)
          link.push_back(corner);
    std::sort(link.begin(), link.end());
    link.erase_till_end(std::unique(link.begin(), link.end()));
    const auto &heads = recovery.row_heads;
    for (const auto v : link) {
      const Index low = std::min(v, site), high = std::max(v, site);
      if (std::size_t(low) >= heads.size())
        continue;
      for (Index k = heads[std::size_t(low)]; k != Index(-1);
           k = recovery.row_links[std::size_t(k)].next) {
        const auto &row = rows[std::size_t(k)];
        const auto segment = std::size_t(row.segment);
        if (std::max(row.a, row.b) != high)
          continue;
        if (recovery.status[segment] != tet_segment_status::recovered)
          return false;
        if ((row.a == site && recovery.first[segment] == k) ||
            (row.b == site && recovery.last[segment] == k))
          continue;
        found.push_back(k);
      }
    }
    if constexpr (tf::core::verification)
      for (std::size_t s = 0; s < recovery.status.size(); ++s) {
        const auto status = recovery.status[s];
        if (status == tet_segment_status::recovered ||
            status == tet_segment_status::retired)
          continue;
        for (Index k = recovery.first[s]; k != Index(-1);
             k = rows[std::size_t(k)].next)
          tf::core::verify(!names_site(rows[std::size_t(k)]),
                           "a site a row not recovered names is not "
                           "collapsed: the census states it past the "
                           "segment phase's fixed point");
      }
    return true;
  };
  if (!collapse_tet_site(owner, site, survivor, consents))
    return false;

  std::size_t joined = 0;
  for (const auto p : found) {
    auto &before = rows[std::size_t(p)];
    if (before.b != site)
      continue;
    const Index q = before.next;
    auto &after = rows[std::size_t(q)];
    if constexpr (tf::core::verification)
      tf::core::verify(after.a == site, "a row ending at a site that is no "
                                        "segment end runs on from it");
    withdraw_tet_subsegment(recovery, std::min(before.a, before.b), p);
    withdraw_tet_subsegment(recovery, std::min(after.a, after.b), q);
    ++joined;
    if (before.a == survivor || after.b == survivor) {
      auto &last = recovery.last[std::size_t(before.segment)];
      before.b = after.b;
      before.n1 = after.n1;
      before.next = after.next;
      if (last == q)
        last = p;
      ticket_tet_subsegment(recovery, p);
      continue;
    }
    before.b = survivor;
    after.a = survivor;
    ticket_tet_subsegment(recovery, p);
    ticket_tet_subsegment(recovery, q);
  }
  if constexpr (tf::core::verification)
    tf::core::verify(2 * joined == found.size(),
                     "every row at the site is one of its segment's two");
  return true;
}

} // namespace tf::topology::cdt::dt3
