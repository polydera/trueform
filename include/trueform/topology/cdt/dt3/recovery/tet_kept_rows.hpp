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
#include "../../../../core/range.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Which recorded rows a restoration seeds from: every live one where no
/// `winding` is given, and where one is, the finite rows it winds positively,
/// one winding per finite row. A restoration that follows the region labeling
/// takes its winding lane, one that precedes it takes none, so both read
/// through one text.
template <typename Index> struct tet_kept_rows {
  tf::range<const int *, tf::dynamic_size> winding{nullptr, nullptr};
  bool given = false;

  auto operator()(Index row) const -> bool {
    return !given ||
           (std::size_t(row) < winding.size() && winding[std::size_t(row)] > 0);
  }
};

} // namespace tf::topology::cdt::dt3
