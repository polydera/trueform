/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/range.hpp"

namespace tf {

template <typename Range>
struct preserve_regions_t {
  Range face_regions;
};

template <typename R>
auto preserve_regions(R &&r) {
  auto wrapped = tf::make_range(static_cast<R &&>(r));
  return preserve_regions_t<decltype(wrapped)>{wrapped};
}

} // namespace tf
