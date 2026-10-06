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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_site_candidate.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// First proposal coincident with a standing site or an earlier proposal;
/// proposals.size() when every proposed position is new.
template <typename Sites, typename Proposals, typename Coord>
auto find_tet_site_coincidence(const Sites &sites, const Proposals &proposals,
                               tf::buffer<tet_site_candidate<Coord>> &records)
    -> std::size_t {
  const std::size_t n = sites.size(), m = proposals.size();
  if (!m)
    return m;
  records.allocate(n + m);
  tf::parallel_for_each(
      tf::make_sequence_range(n + m),
      [&](std::size_t i) {
        if (i < n)
          records[i] = {sites[i].pt, 0};
        else
          records[i] = {proposals[i - n].pt, i - n + 1};
      },
      tf::checked);
  tbb::parallel_sort(
      records.begin(), records.end(), [](const auto &a, const auto &b) {
        return a.pt < b.pt || (a.pt == b.pt && a.ticket < b.ticket);
      });
  std::size_t first = m;
  for (std::size_t i = 1; i < records.size(); ++i)
    if (records[i].ticket && records[i].pt == records[i - 1].pt)
      first = std::min(first, records[i].ticket - 1);
  return first;
}

} // namespace tf::topology::cdt::dt3
