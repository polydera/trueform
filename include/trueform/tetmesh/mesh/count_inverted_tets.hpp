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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/checked.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/tetrahedra_buffer.hpp"
#include "../../core/views/mapped_range.hpp"
#include <cstddef>
#include <functional>

namespace tf::tetmesh {

/// The tetrahedra of `mesh` whose volume, read in its own coordinate type,
/// is negative.
template <typename Index, typename RealT>
auto count_inverted_tets(const tf::tetrahedra_buffer<Index, RealT> &mesh)
    -> std::size_t {
  return tf::reduce(
      tf::make_mapped_range(mesh.tetrahedra(),
                            [](const auto &tet) {
                              const auto a = tet[0];
                              return std::size_t(
                                  tf::dot(tet[1] - a,
                                          tf::cross(tet[2] - a, tet[3] - a)) <
                                  RealT(0));
                            }),
      std::plus<>{}, std::size_t(0), tf::checked);
}

} // namespace tf::tetmesh
