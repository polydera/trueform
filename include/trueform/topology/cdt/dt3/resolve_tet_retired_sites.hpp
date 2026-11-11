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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Rename every input of the index map onto `standing(site)`, the site its
/// slot speaks after a statement's collapses, as the caller's identity
/// authority answers it: the map's one pass over the inputs, once after the
/// statement however many collapses it held.
template <typename Owner, typename Standing>
auto resolve_tet_retired_sites(Owner &owner, const Standing &standing)
    -> void {
  auto &f = owner._index_map.f();
  tf::parallel_for_each(
      tf::make_sequence_range(f.size()),
      [&f, &standing](std::size_t input) { f[input] = standing(f[input]); },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
