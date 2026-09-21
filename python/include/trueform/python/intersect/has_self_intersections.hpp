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
#include "../arrangement/arrangement_builders.hpp"
#include "../spatial/mesh.hpp"
#include "./build_intersect_structures.hpp"
#include <cstddef>
#include <trueform/core/transformation.hpp>
#include <trueform/intersect/has_self_intersections.hpp>

namespace tf::py {
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto has_self_intersections(mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper)
    -> bool {
  build_intersect_structures(form_wrapper);
  // The verdict runs in the mesh's own coordinates, independent of the
  // wrapper's transformation.
  return tf::has_self_intersections(tagged_form(
      form_wrapper, tf::make_identity_transformation<RealT, Dims>()));
}
} // namespace tf::py
