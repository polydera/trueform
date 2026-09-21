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

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/intersect/has_self_intersections.hpp"
#include "trueform/intersect/has_self_intersections.hpp"

#include <cstddef>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Ngon>
auto has_self_intersections(const mesh<Index, Real, 3, Ngon> &value) -> bool {
  if (value.number_of_faces() == 0)
    return false;
  return tf::has_self_intersections(value.topology_form());
}

} // namespace tf::cpp
