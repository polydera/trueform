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
#include "./tet_claim_stream.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows a stream asks to hold: its sites left at `density` rows each,
/// with the margin and the floor, and at least the rows the cavity that
/// stopped it takes; none once it holds no site.
template <typename Index>
auto tet_claim_rows_asked(const tet_claim_stream<Index> &stream, double density)
    -> std::size_t {
  using Tuning = tet_execution_tuning;
  if (stream.begin == stream.end)
    return 0;
  const auto sites = std::size_t(std::ceil(Tuning::asked_rows_margin * density *
                                           double(stream.end - stream.begin)));
  return std::max(sites + Tuning::asked_rows_floor, stream.stopping);
}

} // namespace tf::topology::cdt::dt3
