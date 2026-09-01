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

#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_recovery_statement.hpp"
#include "tbb/parallel_sort.h"

#include <cstddef>
#include <tuple>

namespace tf::arrangement {

/// Equal exact PA names are one identity class. The parameter participates in
/// ordering only for refinement names; stock names retain the LA ordering
/// because their parameter is zero.
template <typename Index, typename Param>
auto close_plane_recovery_names(
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<Index> &class_offsets) -> Index {
  class_offsets.clear();
  if (statements.size() == 0)
    return Index(0);
  tbb::parallel_sort(statements.begin(), statements.end(),
                     [](const plane_recovery_statement<Index, Param> &x,
                        const plane_recovery_statement<Index, Param> &y) {
                       return std::tie(x.name, x.tier, x.root, x.plane) <
                              std::tie(y.name, y.tier, y.root, y.plane);
                     });
  class_offsets.reserve(statements.size() + 1);
  tf::sequenced_generate(
      tf::make_sequence_range(statements.size()), class_offsets,
      [&statements](std::size_t at, tf::buffer<Index> &out) {
        if (at == 0 || !(statements[at - 1].name == statements[at].name))
          out.push_back(Index(at));
      },
      tf::checked);
  class_offsets.push_back(Index(statements.size()));
  return Index(class_offsets.size()) - Index(1);
}

} // namespace tf::arrangement
