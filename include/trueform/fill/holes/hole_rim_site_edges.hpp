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
#include "../../core/buffer.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

template <typename Index>
auto hole_rim_site_edges(const tf::buffer<Index> &site_of_position) {
  const auto n = site_of_position.size();
  return tf::make_mapped_range(
      tf::make_sequence_range(n), [&site_of_position, n](std::size_t i) {
        return std::array<Index, 2>{site_of_position[i],
                                    site_of_position[i + 1 == n ? 0 : i + 1]};
      });
}

} // namespace tf::fill
