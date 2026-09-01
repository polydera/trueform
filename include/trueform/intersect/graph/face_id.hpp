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

#include <tuple>

namespace tf::intersect::graph {

template <typename Index> struct face_id {
  short tag;
  Index object;

  auto operator<(const face_id &o) const {
    return std::tie(tag, object) < std::tie(o.tag, o.object);
  }
  auto operator==(const face_id &o) const {
    return tag == o.tag && object == o.object;
  }
};

} // namespace tf::intersect::graph
