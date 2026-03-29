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

#include "../core/algorithm/circular_increment.hpp"
#include "./topo_id.hpp"

namespace tf {

template <typename Index>
auto is_on_same_edge(const tf::topo_id<Index> &t0, const tf::topo_id<Index> &t1,
                     std::size_t face_size) -> bool {
  auto v = tf::topo_type::vertex;
  auto e = tf::topo_type::edge;
  if (t0.label == v && t1.label == v)
    return t1.id == tf::circular_increment(t0.id, Index(face_size)) ||
           t0.id == tf::circular_increment(t1.id, Index(face_size));
  if (t0.label == e && t1.label == e)
    return t0.id == t1.id;
  if (t0.label == v && t1.label == e)
    return t0.id == t1.id ||
           t0.id == tf::circular_increment(t1.id, Index(face_size));
  if (t0.label == e && t1.label == v)
    return t1.id == t0.id ||
           t1.id == tf::circular_increment(t0.id, Index(face_size));
  return false;
}

} // namespace tf
