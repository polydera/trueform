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
#include "../../core/point.hpp"

namespace tf::arrangement::reseat {

/// The carriers one round re-seats, in the operands' current names: an
/// edge `(tag, a, b)` with `a < b`, split at `point`, found on `face`, and
/// a face `(tag, face)`, fanned at `point`. Both are sorted and unique, an
/// edge by its ends and point, keeping the smallest face that found it.
template <typename Index, typename Int> struct reseat_targets {
  struct edge_t {
    Index tag;
    Index a;
    Index b;
    tf::point<Int, 3> point;
    Index face;
  };
  struct face_t {
    Index tag;
    Index face;
    tf::point<Int, 3> point;
  };
  tf::buffer<edge_t> edges;
  tf::buffer<face_t> faces;

  auto size() const -> std::size_t { return edges.size() + faces.size(); }
};

} // namespace tf::arrangement::reseat
