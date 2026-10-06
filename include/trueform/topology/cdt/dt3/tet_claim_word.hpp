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
#include <cstdint>
#include <limits>

namespace tf::topology::cdt::dt3 {

using tet_claim_word = std::uint32_t;
inline constexpr tet_claim_word tet_unowned =
    std::numeric_limits<tet_claim_word>::max();
inline constexpr tet_claim_word tet_pooled =
    tet_claim_word(std::uint32_t(1) << (sizeof(tet_claim_word) * 8 - 1));
inline constexpr tet_claim_word tet_conflict = tet_pooled >> 1;
inline constexpr tet_claim_word tet_classified = tet_conflict >> 1;
inline constexpr tet_claim_word tet_worker_mask = tet_classified - 1;

} // namespace tf::topology::cdt::dt3
