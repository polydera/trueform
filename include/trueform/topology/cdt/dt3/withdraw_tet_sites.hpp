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
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Remove the sites `_withdrawn` names, ascending slots of the batch that
/// began at `first` and that no cell holds, closing up the batch's slots in
/// order. A withdrawn site's input slot maps to `none`. The remap is
/// monotone, so the canonical cell order and slot order hold, and when the
/// withdrawn sites are the batch's tail no corner moves.
template <typename Owner>
auto withdraw_tet_sites(Owner &owner, std::size_t first, std::size_t input)
    -> void {
  using Index = typename Owner::index_type;
  const auto &withdrawn = owner._withdrawn;
  const auto closed = [&withdrawn](Index slot) {
    return Index(
        std::size_t(slot) -
        std::size_t(std::lower_bound(withdrawn.begin(), withdrawn.end(), slot) -
                    withdrawn.begin()));
  };
  const Index from = Index(first);
  const bool tail =
      withdrawn.size() == owner._floors.size() - std::size_t(withdrawn[0]);
  if (!tail)
    tf::parallel_for_each(
        tf::make_sequence_range(owner._corners.size()),
        [&owner, &closed, from](std::size_t cell) {
          for (auto &corner : owner._corners[cell])
            if (corner >= from)
              corner = closed(corner);
        },
        tf::checked);

  const std::size_t count = owner._floors.size() - first;
  std::size_t kept = first;
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t slot = first + i;
    if (std::binary_search(withdrawn.begin(), withdrawn.end(), Index(slot))) {
      owner._index_map.f()[input + i] = Owner::none;
      continue;
    }
    owner._floors[kept] = owner._floors[slot];
    if (owner._record_site_cells)
      owner._site_cells.store(kept, owner._site_cells.load(slot));
    owner._index_map.f()[input + i] = Index(kept);
    owner._index_map.kept_ids()[kept] = Index(input + i);
    ++kept;
  }
  owner._floors.allocate(kept);
  if (owner._record_site_cells)
    owner._site_cells.resize(kept, Owner::none);
  owner._index_map.kept_ids().allocate(kept);
}

} // namespace tf::topology::cdt::dt3
