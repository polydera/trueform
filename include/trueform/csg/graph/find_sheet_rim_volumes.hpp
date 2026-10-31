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
#include "../../core/range.hpp"
#include "./domain_inclusions.hpp"
#include "./sheet_sides.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The `(sheet tag, volume tag)` pairs whose volume holds a free rim
///        of the sheet, ascending.
///
/// A rim (@ref tf::csg::graph::find_sheet_rims) is interior to a volume
/// exactly when both of its sides lie inside it; one side out puts the
/// component on the volume's own wall. Reads the volume columns of `inc`,
/// which are final.
template <typename Index>
auto find_sheet_rim_volumes(const tf::csg::graph::sheet_sides<Index> &sides,
                            const tf::buffer<std::array<Index, 3>> &rims,
                            const tf::csg::graph::domain_inclusions &inc)
    -> tf::buffer<std::array<Index, 2>> {
  tf::buffer<std::array<Index, 2>> out;
  const Index n_tags = Index(sides.sheet_of_tag.size());
  tf::generic_generate(
      tf::make_range(rims), out,
      [&](const std::array<Index, 3> &rim,
          tf::buffer<std::array<Index, 2>> &buffer) {
        for (Index t = Index(0); t < n_tags; ++t)
          if (sides.sheet_of_tag[t] < Index(0) &&
              inc.test(std::size_t(rim[1]), std::size_t(t)) &&
              inc.test(std::size_t(rim[2]), std::size_t(t)))
            buffer.push_back({sides.tags[rim[0]], t});
      },
      tf::checked);
  tbb::parallel_sort(out.begin(), out.end());
  out.erase_till_end(std::unique(out.begin(), out.end()));
  return out;
}

} // namespace tf::csg::graph
