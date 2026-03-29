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
#include "./boolean.hpp"

namespace tf::cut::dispatch {

template <typename Policy, typename F>
auto self_boolean(const tf::polygons<Policy> &form, F &&f) {
  using S = decltype(make_missing_structures(form));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return f(form);
  } else {
    auto s = make_missing_structures(form);
    return f(tag_with_structures(form, s));
  }
}

} // namespace tf::cut::dispatch
