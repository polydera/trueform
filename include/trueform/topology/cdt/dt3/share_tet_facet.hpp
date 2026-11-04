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
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Add `face` to the stack of the shared facet `key`, once: the row joins
/// the table in its sorted place.
template <typename Owner>
auto share_tet_facet(Owner &owner,
                     const std::array<typename Owner::index_type, 3> &key,
                     typename Owner::index_type face) -> void {
  auto &shared = owner._shared;
  const std::pair<std::array<typename Owner::index_type, 3>,
                  typename Owner::index_type>
      row{key, face};
  const auto at = std::lower_bound(shared.begin(), shared.end(), row);
  if (at != shared.end() && *at == row)
    return;
  const auto offset = std::ptrdiff_t(at - shared.begin());
  shared.push_back(row);
  std::rotate(shared.begin() + offset, shared.end() - 1, shared.end());
}

} // namespace tf::topology::cdt::dt3
