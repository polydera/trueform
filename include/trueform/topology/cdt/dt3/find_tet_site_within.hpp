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
#include "../../../core/buffer.hpp"
#include "./for_each_tet_site_within.hpp"
#include "./tet_position.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The smallest site `admits` names in `p`'s cell of the reach grid
/// (@ref for_each_tet_site_within), `k_none` where none is. O(the cells
/// meeting the box of `reach`).
template <typename DT, typename Admits, typename Take>
auto find_tet_site_within(const DT &dt,
                          const tet_position<typename DT::coord_type> &p,
                          typename DT::index_type cell,
                          typename DT::coord_type reach, const Admits &admits,
                          const Take &take,
                          tf::buffer<typename DT::index_type> &flood) ->
    typename DT::index_type {
  auto found = DT::k_none;
  for_each_tet_site_within(
      dt, p, cell, reach,
      [&found, &admits](typename DT::index_type site) {
        if ((found == DT::k_none || site < found) && admits(site))
          found = site;
      },
      take, flood);
  return found;
}

} // namespace tf::topology::cdt::dt3
