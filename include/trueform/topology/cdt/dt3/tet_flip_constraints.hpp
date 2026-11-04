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
#include "../../../core/range.hpp"
#include "./tet_lifted_change.hpp"
#include <array>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// What a flip may not do, and what its caller wants kept: a flip is
/// refused when a new edge or facet would cross `segment` (its two sites),
/// when a new edge would pierce `polygon` (a face's triangles, the first
/// naming its plane, read on `axes`), or when it would hold `removed`,
/// each `none` or empty when not given (@ref tet_flip_eligible); `unflip`
/// undoes a failed removal's flips and `collect` lists the cells a removal
/// made (@ref flip_tet_ring); `collect_encroached` lists the segments a
/// removal met; where `lifted` is given every flip adds its change of the
/// lifted volume to it (@ref tet_lifted_change) and a removal that does not
/// lower it is undone.
template <typename Index, typename Coord, typename Int>
struct tet_flip_constraints {
  Index segment[2] = {Index(-1), Index(-1)};
  tf::range<const std::array<Index, 3> *, tf::dynamic_size> polygon{nullptr,
                                                                     nullptr};
  std::pair<int, int> axes{0, 1};
  Index removed = Index(-1);
  bool unflip = false;
  bool collect = false;
  bool collect_encroached = false;
  tet_lifted_change<Int, Coord> *lifted = nullptr;
};

} // namespace tf::topology::cdt::dt3
