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

/// An appended site's Morton key beside its slot
/// (@ref order_appended_tet_sites).
template <typename Index> struct tet_appended_key {
  std::uint64_t key;
  Index site;
};

} // namespace tf::topology::cdt::dt3
