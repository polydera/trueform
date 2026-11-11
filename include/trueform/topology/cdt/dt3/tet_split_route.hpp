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

namespace tf::topology::cdt::dt3 {

/// The route an append states to place its sites by the split of the
/// simplex each lies in, whatever the complex
/// (@ref insert_tetrahedralization_site): a point a recovery places on a
/// plane whose facets its lane holds is no Delaunay insertion, since its
/// conflict region reaches across a held facet and wraps it.
struct tet_split_route_t {};
inline constexpr tet_split_route_t tet_split_route{};

} // namespace tf::topology::cdt::dt3
