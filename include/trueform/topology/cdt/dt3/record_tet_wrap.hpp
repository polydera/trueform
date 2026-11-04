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
#include "./tet_facet_key.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// State that a cavity wrapped the protected facet of `cell` opposite
/// `slot`.
template <typename Owner>
auto record_tet_wrap(Owner &owner, typename Owner::index_type cell,
                     std::size_t slot) -> void {
  owner._wraps.push_back(
      tet_facet_key(owner._corners[std::size_t(cell)], slot));
}

} // namespace tf::topology::cdt::dt3
