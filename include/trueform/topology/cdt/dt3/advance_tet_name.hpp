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

/// The first fresh name once a site carries `name`, from the first fresh
/// name `next` before it. A negative name lies below every fresh one.
template <typename Index>
auto advance_tet_name(std::size_t next, Index name) -> std::size_t {
  return name < Index(0) ? next : std::max(next, std::size_t(name) + 1);
}

} // namespace tf::topology::cdt::dt3
