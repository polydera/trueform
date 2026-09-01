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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../intersect/graph/plane_tables.hpp"

namespace tf::arrangement {

/// The carriers of a set of world groups: the plane of every face stating an
/// instance of one. A plane appears once per instance it holds, because the
/// caller that asks closes the set the way its own diff needs it.
template <typename Index, typename Int, typename Groups, typename PlaneOfFace>
auto state_plane_group_carriers(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const Groups &groups, const PlaneOfFace &plane_of_face,
    tf::buffer<Index> &carriers) -> void {
  tf::generic_generate(
      groups, carriers,
      [&](Index group, tf::buffer<Index> &out) {
        for (const auto &def : world_tables.canon_group(group))
          out.push_back(plane_of_face(def.face));
      },
      tf::checked);
}

} // namespace tf::arrangement
