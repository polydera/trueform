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

namespace tf::intersect::graph {

/// Per-face descriptor storing the mesh tag and face object index.
template <typename Index> struct face_descriptor {
  Index tag;
  Index object;
};

} // namespace tf::intersect::graph
