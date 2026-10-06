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
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_site_key.hpp"
#include "./tet_site_priority.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Rank every site by its process key into the order lane, the ranking the
/// insertion bands are cut from.
template <bool Parallel, typename Owner>
auto rank_tet_sites(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._sites.size();
  owner._priorities.allocate(n);
  owner._order.allocate(n);
  owner._key_records.allocate(n);
  const auto key_site = [&owner](std::size_t i) {
    const auto priority = tet_site_priority(std::uint64_t(owner._sites[i].id));
    owner._priorities[i] = priority;
    owner._key_records[i] = {priority, Index(i)};
  };
  const auto take_site = [&owner](std::size_t i) {
    owner._order[i] = owner._key_records[i].site;
  };
  const auto less = [](const tet_site_key<Index> &a,
                       const tet_site_key<Index> &b) {
    return a.code < b.code || (a.code == b.code && a.site < b.site);
  };
  if constexpr (Parallel) {
    tf::parallel_for_each(tf::make_sequence_range(n), key_site, tf::checked);
    tbb::parallel_sort(owner._key_records.begin(), owner._key_records.end(),
                       less);
    tf::parallel_for_each(tf::make_sequence_range(n), take_site, tf::checked);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      key_site(i);
    std::sort(owner._key_records.begin(), owner._key_records.end(), less);
    for (std::size_t i = 0; i < n; ++i)
      take_site(i);
  }
}

} // namespace tf::topology::cdt::dt3
