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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The protected facets, held or not; zero is the stock complex.
template <typename Owner>
auto tet_protected_count(const Owner &owner) -> std::size_t {
  return owner._n_held + owner._protected_unheld.size();
}

} // namespace tf::topology::cdt::dt3
