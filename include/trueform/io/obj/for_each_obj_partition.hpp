/*
 * Copyright (c) 2025 XLAB
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

#include "../../core/algorithm/isolated_parallel_for_each.hpp"
#include "../../core/views/sequence_range.hpp"

#include <cstddef>

namespace tf::io::obj {

template <typename Function>
auto for_each_obj_partition(std::size_t count, Function &&function) -> void {
  if (count == 1) {
    function(0);
    return;
  }
  tf::parallel_for_each(tf::make_sequence_range(count),
                        static_cast<Function &&>(function), tf::isolated);
}

} // namespace tf::io::obj
