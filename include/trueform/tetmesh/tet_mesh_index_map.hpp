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
#include "../core/offset_block_buffer.hpp"
#include <array>

namespace tf {

/// @ingroup tetmesh
/// @brief Where the input surface of a @ref tf::make_tet_mesh call stands
///        in its tet mesh.
///
/// Block `k` of `facet_triangles` holds the boundary triangles of the tet
/// mesh covering facet `k`, indexed into its points and wound as the facet
/// is; block `f` of `face_facets` lists, ascending, the facets input face
/// `f` lies in, one unless a crossing cut it.
template <typename Index> struct tet_mesh_index_map {
  tf::offset_block_buffer<Index, std::array<Index, 3>> facet_triangles;
  tf::offset_block_buffer<Index, Index> face_facets;
};

} // namespace tf
