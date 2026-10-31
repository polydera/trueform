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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./sheet_sides.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The free rims of the sheets as `(sheet, side 0, side 1)`, the two
///        domains an open component carrying a layer of the sheet stands
///        between, smaller first, ascending.
template <typename Index, typename OpenMask>
auto find_sheet_rims(const tf::csg::graph::sheet_sides<Index> &sides,
                     const tf::csg::graph::arrangement_descriptor<Index> &desc,
                     const tf::csg::graph::component_crossings<Index> &crossings,
                     const OpenMask &open_mask)
    -> tf::buffer<std::array<Index, 3>> {
  tf::buffer<std::array<Index, 3>> rims;
  if (sides.n_sheets() == Index(0))
    return rims;
  tf::generic_generate(
      tf::make_sequence_range(Index(desc.tag_of_component.size())), rims,
      [&](Index c, tf::buffer<std::array<Index, 3>> &buffer) {
        if (!open_mask[c])
          return;
        const Index d0 = desc.domain_of_side[2 * c + 0];
        const Index d1 = desc.domain_of_side[2 * c + 1];
        sides.for_each_sheet_layer(desc, crossings, c, [&](Index k, Index) {
          buffer.push_back({k, std::min(d0, d1), std::max(d0, d1)});
        });
      },
      tf::checked);
  tbb::parallel_sort(rims.begin(), rims.end());
  rims.erase_till_end(std::unique(rims.begin(), rims.end()));
  return rims;
}

} // namespace tf::csg::graph
