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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index> struct tet_claim_scratch {
  tf::buffer<Index> held, cavity, boundary, rows;
  tf::buffer<std::uint8_t> entered;
};

} // namespace tf::topology::cdt::dt3
