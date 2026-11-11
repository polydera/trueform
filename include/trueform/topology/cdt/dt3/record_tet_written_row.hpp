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
#include "./tetrahedralization_owner.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Record that `owner` wrote the finite row `tet`: once per mark, a write of
/// a row the record holds since the last mark (@ref mark_tet_written_rows)
/// counting on its record. The row's ticket into the record is read only
/// where the record still holds it there, so a clear or a mark needs no
/// sweep of the tickets.
template <typename Owner>
auto record_tet_written_row(Owner &owner, typename Owner::index_type tet)
    -> void {
  using Index = typename Owner::index_type;
  auto &at = owner._written_at;
  if (std::size_t(tet) >= at.size()) {
    const std::size_t known = at.size();
    at.reallocate(std::max(std::size_t(tet) + 1, 2 * known));
    std::fill(at.begin() + std::ptrdiff_t(known), at.end(), Owner::none);
  }
  auto &written = owner._written;
  const Index k = at[std::size_t(tet)];
  if (k != Owner::none && std::size_t(k) >= owner._written_floor &&
      std::size_t(k) < written.size() && written[std::size_t(k)].row == tet) {
    ++written[std::size_t(k)].writes;
    return;
  }
  at[std::size_t(tet)] = Index(written.size());
  written.push_back({tet, 1u});
}

} // namespace tf::topology::cdt::dt3
