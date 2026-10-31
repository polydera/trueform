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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"

#include <array>
#include <cstddef>
#include <tuple>

namespace tf::arrangement {

/// THE WELD'S LAW read off a closed layer (@ref identity_substrate): one key
/// holds one target, so the keys ascend strictly; every target is `is_live`
/// — retired by nothing — in what the layer was stated into; and, under a
/// currency of `n_created` created identities, every target stands inside
/// it — `-1` states none. A verification check; its caller states that.
template <typename Index, typename IsLive>
auto assert_welds_are_lawful(const tf::buffer<std::array<Index, 3>> &rows,
                             Index n_created, const IsLive &is_live) -> void {
  tf::parallel_for_each(
      tf::make_sequence_range(rows.size()),
      [&](std::size_t at) {
        const auto &row = rows[at];
        tf::core::verify(at == 0 || std::tie(rows[at - 1][0], rows[at - 1][1]) <
                                        std::tie(row[0], row[1]),
                         "a weld's key holds one target");
        tf::core::verify(n_created == Index(-1) || row[2] < n_created,
                         "a weld's target speaks the currency of its layer");
        tf::core::verify(is_live(row[2]), "a weld's target is live");
      },
      tf::checked);
}

} // namespace tf::arrangement
