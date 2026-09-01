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
#include "../../arrangement/mesh/mesh_triangulation.hpp"
#include "../../core/algorithm/parallel_copy_blocked.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/range.hpp"
#include <cstddef>

namespace tf::geometry {

/// The triangles a mesh triangulation states, as the face buffer an entry
/// point returns.
///
/// THE CORNERS NEED NO REMAP. The identity space is the input's, so a corner
/// below `n_original_points()` IS the input vertex id and one at or past it is
/// the row of `created_points()` that follows the originals — which is exactly
/// where @ref tf::arrangement::materialize_mesh_triangulation puts it.
template <typename Index, typename RealT, typename Int, std::size_t Dims,
          typename Faces>
auto make_triangulation_faces(
    const tf::arrangement::mesh_triangulation<Index, RealT, Int, Dims, Faces>
        &triangulation) -> tf::blocked_buffer<Index, 3> {
  const auto triangles = triangulation.triangles();
  tf::blocked_buffer<Index, 3> faces;
  faces.allocate(triangles.size());
  tf::parallel_copy_blocked(triangles, tf::make_range(faces));
  return faces;
}

} // namespace tf::geometry
