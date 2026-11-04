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
#include "./mark_tet_cavity_edges.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_cavity_links_hold.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Killed, typename Index>
auto keep_taken_tet_edges(Killed &killed, std::size_t first,
                          tet_cavity_links<Index> &links) -> void {
  if (!links.overflowed)
    mark_tet_cavity_edges(links, tet_cavity_links<Index>::marked);
  std::size_t kept = first;
  for (std::size_t i = first; i < killed.size(); ++i)
    if (!tet_cavity_links_hold(links, killed[i][0], killed[i][1]))
      killed[kept++] = killed[i];
  killed.erase_till_end(killed.begin() + std::ptrdiff_t(kept));
  if (!links.overflowed)
    mark_tet_cavity_edges(links, Index(-1));
}

} // namespace tf::topology::cdt::dt3
