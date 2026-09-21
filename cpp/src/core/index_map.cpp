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
#include "trueform/cpp/core/index_map.hpp"

#include <cstdint>
#include <utility>

namespace tf::cpp {

template <typename Index>
auto index_map<Index>::from_index_map_buffer(
    tf::index_map_buffer<Index> &&buffer) -> index_map {
  return {nd_array<Index>::from_buffer(std::move(buffer.f())),
          nd_array<Index>::from_buffer(std::move(buffer.kept_ids()))};
}

template <typename Index>
auto index_map<Index>::deep_copy() const -> index_map {
  return {f.deep_copy(), kept_ids.deep_copy()};
}

template <typename Index> auto index_map<Index>::is_valid() const -> bool {
  return f.is_valid() && kept_ids.is_valid();
}

template struct index_map<std::int32_t>;
template struct index_map<std::int64_t>;

} // namespace tf::cpp
