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

#include "../../exact/vertex.hpp"
#include "../../topology/topo_id.hpp"

namespace tf::exact {

/// Push an intersection record + point into the thread-local buffers.
template <typename Index, typename Ints, typename Pts>
auto emit_record(int tag, int tag_other, Index object, Index object_other,
                 tf::topo_id<Index> target, tf::topo_id<Index> target_other,
                 const pt3 &point, Ints &ints, Pts &pts) {
  Index id = pts.size();
  pts.push_back(point);
  ints.push_back({Index(tag), Index(tag_other), object, object_other, target,
                  target_other, id});
}

} // namespace tf::exact
