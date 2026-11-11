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
#include "../../tet_claim_word.hpp"
#include <atomic>

namespace tf::topology::cdt::dt3 {

/// Lower `word` to `value` where it holds more, `tet_unowned` holding more
/// than every value: concurrent lowerings leave the smallest, whatever
/// their order.
inline auto lower_tet_claim_word(std::atomic<tet_claim_word> &word,
                                 tet_claim_word value) -> void {
  tet_claim_word held = word.load(std::memory_order_relaxed);
  while (value < held &&
         !word.compare_exchange_weak(held, value, std::memory_order_relaxed))
    ;
}

} // namespace tf::topology::cdt::dt3
