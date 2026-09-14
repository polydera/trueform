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

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/spatial/distance.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Compute signed distances from a 3D mesh to a point or batch.
///
/// The magnitude is the distance to the closest point of the surface; the sign
/// is the generalized winding number, negative inside and positive outside,
/// which is the graceful answer where a mesh is open or a soup. Query
/// coordinates are converted to the form precision before computation, and the
/// query is a point: any other kind is refused. Empty query batches return an
/// empty batch without reading the form. A scalar or nonempty query against a
/// mesh with no faces throws `std::invalid_argument`.
/// @note Reads the TREE and the WINDING MOMENTS.
template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal, std::enable_if_t<Dims == 3, int> = 0>
auto signed_distance(const mesh<Index, FormReal, Dims, Ngon> &form,
                     const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal>;

} // namespace tf::cpp
