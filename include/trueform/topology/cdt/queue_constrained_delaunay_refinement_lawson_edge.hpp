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

namespace tf::topology::cdt {

template <typename Owner>
auto queue_constrained_delaunay_refinement_lawson_edge(
    Owner &owner, typename Owner::index_type face, int edge) -> void {
  owner._lawson.push_back((face << 2) | edge);
}

} // namespace tf::topology::cdt
