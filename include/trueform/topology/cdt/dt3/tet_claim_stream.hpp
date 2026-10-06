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
#include "../../../core/cache_aligned_slot.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Index>
struct alignas(tf::core::cache_aligned_slot<int>::alignment) tet_claim_stream {
  std::size_t begin = 0, end = 0, free_count = 0;
  std::size_t fresh_begin = 0, fresh_end = 0;
  Index free_head = -1, hint_begin = -1, hint_end = -1;
};

} // namespace tf::topology::cdt::dt3
