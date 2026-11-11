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
#include "../../../core/config/verify.hpp"
#include "./tet_edge_ref.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The ticket `4 cell + slot` of the facet a handle reads as its edge and
/// apex, named from the lower of the facet's two cells.
template <typename Owner>
auto tet_facet_ticket(const Owner &owner,
                      const tet_edge_ref<typename Owner::index_type> &handle) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index cell = handle.cell;
  const std::size_t slot = tet_edge_oppo(handle);
  const Index across = owner._neighbors[std::size_t(cell)][slot];
  if (across < cell) {
    std::size_t back = 0;
    while (back < 4 && owner._neighbors[std::size_t(across)][back] != cell)
      ++back;
    if constexpr (tf::core::verification)
      tf::core::verify(back < 4, "the mate names the cell back");
    return Index(4 * std::size_t(across) + back);
  }
  return Index(4 * std::size_t(cell) + slot);
}

} // namespace tf::topology::cdt::dt3
