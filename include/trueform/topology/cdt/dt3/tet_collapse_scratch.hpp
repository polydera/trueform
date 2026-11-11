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
#include "../../../core/buffer.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The lists one collapse of a site walks on (@ref collapse_tet_site):
/// `star` the cells holding the site, admitted once through `marks`, a
/// generation per row that `generation` advances each collapse; `ring` the
/// cells holding the survivor too; `edges` the site's edges it takes; and
/// `faces`, the collapse's product, the faces whose held facets it moved or
/// removed, ascending, which the face phase reads again.
template <typename Index> struct tet_collapse_scratch {
  tf::buffer<Index> star;
  tf::buffer<std::uint32_t> marks;
  std::uint32_t generation = 0;
  tf::buffer<Index> ring;
  tf::buffer<std::array<Index, 2>> edges;
  tf::buffer<Index> faces;
};

} // namespace tf::topology::cdt::dt3
