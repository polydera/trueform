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

#include "../../core/buffer.hpp"

#include <cstddef>

namespace tf::io::obj {

inline auto make_obj_line_partitions(const char *data, std::size_t size,
                                     std::size_t count)
    -> tf::buffer<std::size_t> {
  tf::buffer<std::size_t> boundaries;
  boundaries.allocate(count + 1);
  boundaries.front() = 0;
  boundaries.back() = size;
  for (std::size_t partition = 1; partition < count; ++partition) {
    auto offset =
        (size / count) * partition + ((size % count) * partition) / count;
    while (offset < size && data[offset - 1] != '\n' &&
           data[offset - 1] != '\r')
      ++offset;
    while (offset < size && (data[offset] == '\n' || data[offset] == '\r'))
      ++offset;
    boundaries[partition] = offset;
  }
  return boundaries;
}

} // namespace tf::io::obj
