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
#include "../../../../core/buffer.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// One mark per key of a dense key space: a mark equal to `generation` says
/// the key was taken this generation, so opening a generation never clears
/// the space.
struct tet_stamps {
  tf::buffer<std::uint32_t> marks;
  std::uint32_t generation = 0;
};

} // namespace tf::topology::cdt::dt3
