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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The row of `rows`, sorted by name, that names `name`, `-1` where none
/// does; a name below the first row's answers at once.
template <typename Index, typename Rows>
auto find_tet_deep_row(const Rows &rows, Index name) -> Index {
  if (!rows.size() || name < rows[0].id)
    return Index(-1);
  const auto at = std::lower_bound(
      rows.begin(), rows.end(), name,
      [](const auto &row, Index value) { return row.id < value; });
  return at != rows.end() && at->id == name ? Index(at - rows.begin())
                                            : Index(-1);
}

} // namespace tf::topology::cdt::dt3
