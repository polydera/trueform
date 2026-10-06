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
#include "./tetra.hpp"
#include <array>
#include <cstddef>

namespace tf::core {

template <typename Policy, typename Transformation>
auto transformed(const tetra<Policy> &tetrahedron,
                 const Transformation &transformation) {
  using pt_t = decltype(transformed(tetrahedron[0], transformation));
  std::array<pt_t, 4> out;
  for (std::size_t i = 0; i < 4; ++i)
    out[i] = transformed(tetrahedron[i], transformation);
  return core::make_tetra(out);
}

} // namespace tf::core
