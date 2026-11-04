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

/// Whether the facet of `cell` opposite `slot` is protected.
template <typename Owner>
auto tet_facet_is_protected(const Owner &owner, typename Owner::index_type cell,
                            std::size_t slot) -> bool {
  const auto &slots = owner._protected_slots;
  return std::size_t(cell) < slots.size() &&
         ((slots[std::size_t(cell)] >> slot) & 1u);
}

} // namespace tf::topology::cdt::dt3
