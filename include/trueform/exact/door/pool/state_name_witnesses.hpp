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

#include "./pool_records.hpp"
#include "./scene_targets.hpp"

#include "../../../core/algorithm/generic_generate.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../placement_tables.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::exact::door::pool {

/// The witnesses: one per face that speaks for its own exact plane, holding
/// the names that must all resolve to that plane for the face to be realized
/// on it, and the corners themselves.
///
/// A face whose every corner is flat and has a target source is a witness; a
/// face carrying a feature corner speaks for nothing, because the rank-3/2
/// cascade owns that corner's position.
///
/// The support triple is sorted, so a face names one witness whatever its
/// winding, and the array is sorted so that a name's block comes out in
/// canonical witness order.
template <typename Index, typename Int, typename RealType>
auto state_name_witnesses(const placement_tables<Index, Int, RealType> &tables,
                          const tf::blocked_buffer<Index, 3> &corners,
                          const tf::buffer<int> &face_name,
                          const scene_targets &targets,
                          tf::buffer<pool_witness<Int>> &witness) -> void {
  witness.clear();
  tf::generic_generate(
      tf::make_sequence_range(corners.size()), witness,
      [&tables, &corners, &face_name,
       &targets](std::size_t f, tf::buffer<pool_witness<Int>> &into) {
        if (face_name[f] < 0)
          return;
        const auto block = corners[f];
        for (const auto v : block)
          if (targets.feature[std::size_t(v)] ||
              targets.source[std::size_t(v)] < 0)
            return;

        pool_witness<Int> made;
        made.plane_name = face_name[f];
        for (std::size_t k = 0; k < 3; ++k)
          made.support[k] = tables.points[std::size_t(block[k])];
        std::sort(made.support.begin(), made.support.end());

        std::array<int, 4> stated{
            face_name[f], targets.source[std::size_t(block[0])],
            targets.source[std::size_t(block[1])],
            targets.source[std::size_t(block[2])]};
        std::sort(stated.begin(), stated.end());
        std::size_t kept = 0;
        for (std::size_t k = 0; k < stated.size(); ++k)
          if (k == 0 || stated[k] != stated[k - 1])
            made.name[kept++] = stated[k];
        into.push_back(made);
      });

  tbb::parallel_sort(witness.begin(), witness.end(),
                     [](const pool_witness<Int> &a, const pool_witness<Int> &b) {
                       if (a.plane_name != b.plane_name)
                         return a.plane_name < b.plane_name;
                       if (a.support != b.support)
                         return a.support < b.support;
                       return a.name < b.name;
                     });
}

} // namespace tf::exact::door::pool
