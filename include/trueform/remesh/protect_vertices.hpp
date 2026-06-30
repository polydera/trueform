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

#include "../core/range.hpp"

namespace tf {

template <typename Range>
struct protect_vertices_t {
  Range vertex_mask;
};

template <typename R>
auto protect_vertices(R &&r) {
  auto wrapped = tf::make_range(static_cast<R &&>(r));
  return protect_vertices_t<decltype(wrapped)>{wrapped};
}

} // namespace tf
