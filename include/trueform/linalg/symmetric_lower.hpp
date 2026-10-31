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
#include "../core/buffer.hpp"
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief A symmetric matrix by its lower triangle: row `i` holds the columns
/// `j <= i` it states, ascending, so its diagonal is its last entry.
///
/// An entry's position in `columns` is its ticket: every carrier built from
/// this one names an entry by it.
///
/// @tparam Index The index type rows and entries are numbered in.
template <typename Index> struct symmetric_lower {
  tf::buffer<Index> row_offsets;
  tf::buffer<Index> columns;
  tf::buffer<double> values;

  auto size() const -> std::size_t {
    return row_offsets.size() ? row_offsets.size() - 1 : 0;
  }
};

} // namespace tf
