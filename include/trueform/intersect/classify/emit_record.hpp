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

#include "../../topology/topo_id.hpp"

namespace tf::exact {

/// Push an intersection record + its payload into the caller's local
/// output buffers. The record's id is the payload's slot, so it counts
/// emissions whatever the payload describes.
template <typename Index, typename Intersections, typename Pts,
          typename Payload>
auto emit_record(int tag, int tag_other, Index object, Index object_other,
                 tf::topo_id<Index> target, tf::topo_id<Index> target_other,
                 const Payload &payload, Intersections &intersections,
                 Pts &pts) {
  Index id = pts.size();
  pts.push_back(payload);
  intersections.push_back({short(tag), short(tag_other), object, object_other,
                           target, target_other, id});
}

} // namespace tf::exact
