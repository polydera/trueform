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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/buffer.hpp"
#include <cstddef>

namespace tf::geometry {

/// The faces a mesh triangulation refused: the carriers whose triangulation
/// refused every round of the resolution wave, so they hold no product.
///
/// A CARRIER OF A MESH WORLD IS AN INPUT FACE, so the tier's completeness
/// surface names input faces already — the mapping is the identity, and the
/// ids come out ascending because that surface is a sorted unique set.
template <typename Index, typename RealT, typename Int, std::size_t Dims,
          typename Faces>
auto make_refused_faces(
    const tf::arrangement::mesh_triangulation<Index, RealT, Int, Dims, Faces>
        &triangulation) -> tf::buffer<Index> {
  const auto failed = triangulation.failed();
  tf::buffer<Index> refused;
  refused.allocate(failed.size());
  tf::parallel_copy(failed, refused);
  return refused;
}

} // namespace tf::geometry
