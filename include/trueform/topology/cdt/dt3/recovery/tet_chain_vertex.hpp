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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A vertex of a recovered chain: its site slot and its numerator over
/// 2^depth along the segment, `-1` for a shared site off that grid.
template <typename Index> struct tet_chain_vertex {
  Index site;
  std::int64_t numerator;
};

} // namespace tf::topology::cdt::dt3
