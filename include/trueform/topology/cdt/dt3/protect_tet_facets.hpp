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
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./adopt_tet_facet_owner.hpp"
#include "./share_tet_facet.hpp"
#include "./locate_tet_facet.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_edge_star.hpp"
#include "./tet_facet_is_protected.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Protect facets, each three site slots in any order, as children of
/// `face`, `none` for no face: each is found in the complex and both its
/// cells take the bit and the owner; a facet already protected adopts the
/// face (@ref adopt_tet_facet_owner). A facet the complex does not hold
/// bars nothing; it is kept among the unheld with its owner and counts.
template <typename Owner, typename Facets>
auto protect_tet_facets(Owner &owner, const Facets &facets,
                        typename Owner::index_type face = Owner::none) -> void {
  using Index = typename Owner::index_type;
  tf::buffer<std::array<Index, 3>> keys;
  keys.allocate(facets.size());
  tf::buffer<Index> tickets;
  tickets.allocate(facets.size());
  struct local_t {
    tet_edge_star<Index> star;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(facets.size()),
      [&keys, &tickets, &facets, &owner](std::size_t i, local_t &local) {
        std::array<Index, 3> key{Index(facets[i][0]), Index(facets[i][1]),
                                 Index(facets[i][2])};
        std::sort(key.begin(), key.end());
        keys[i] = key;
        tickets[i] = locate_tet_facet(owner, key, local.star);
      },
      local_t{}, tf::checked);
  auto &unheld = owner._protected_unheld;
  for (std::size_t i = 0; i < facets.size(); ++i) {
    const Index ticket = tickets[i];
    if (ticket == Owner::none) {
      const auto at = std::lower_bound(
          unheld.begin(), unheld.end(), keys[i],
          [](const auto &record, const auto &key) { return record.key < key; });
      if (at != unheld.end() && at->key == keys[i]) {
        if (at->face == Owner::none)
          at->face = face;
        else if (face != Owner::none && at->face != face) {
          if (at->face != Owner::shared_owner) {
            share_tet_facet(owner, keys[i], at->face);
            at->face = Owner::shared_owner;
          }
          share_tet_facet(owner, keys[i], face);
        }
        continue;
      }
      const auto offset = std::ptrdiff_t(at - unheld.begin());
      unheld.push_back({keys[i], face});
      std::rotate(unheld.begin() + offset, unheld.end() - 1, unheld.end());
      continue;
    }
    const Index cell = Index(std::size_t(ticket) / 4);
    const std::size_t slot = std::size_t(ticket) % 4;
    if (tet_facet_is_protected(owner, cell, slot)) {
      adopt_tet_facet_owner(owner, cell, slot, face);
      continue;
    }
    state_tet_facet_protection(owner, cell, slot, true, face);
  }
}

} // namespace tf::topology::cdt::dt3
