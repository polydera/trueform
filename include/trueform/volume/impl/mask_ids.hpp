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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief The sample ids a mask sets, ascending.
inline auto mask_ids(const tf::buffer<char> &mask)
    -> tf::buffer<std::ptrdiff_t> {
  tf::buffer<std::ptrdiff_t> ids;
  tf::sequenced_generate(
      tf::make_sequence_range(std::ptrdiff_t(mask.size())), ids,
      [&](std::ptrdiff_t i, tf::buffer<std::ptrdiff_t> &out) {
        if (mask[std::size_t(i)])
          out.push_back(i);
      },
      tf::checked);
  return ids;
}

} // namespace volume_detail
} // namespace tf
