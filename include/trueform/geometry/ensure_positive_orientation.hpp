/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/polygons.hpp"
#include "../core/signed_volume.hpp"
#include "../topology/orient_faces_consistently.hpp"
#include "../topology/reverse_winding.hpp"

namespace tf {
template <typename Policy>
auto ensure_positive_orientation(tf::polygons<Policy> &polygons,
                                 bool is_consistent = false) -> void {
  if (!is_consistent)
    tf::orient_faces_consistently(polygons);
  auto stripped = tf::make_polygons(polygons.faces(), polygons.points());
  if (tf::signed_volume(stripped) < 0)
    tf::reverse_winding(polygons.faces());
}

template <typename Policy>
auto ensure_positive_orientation(tf::polygons<Policy> &&polygons,
                                 bool is_consistent = false) -> void {
  ensure_positive_orientation(polygons, is_consistent);
}
} // namespace tf
