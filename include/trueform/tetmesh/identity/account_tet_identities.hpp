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
#include "./tet_identity_authority.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::tetmesh {

/// What became of one identity of a drain, read against the sites a
/// published complex names: `live`, standing and named; `retired`, relabeled
/// into another and named by nothing; `vanished`, standing and named by
/// nothing — dead by definition; and the break, `emitted_dead`, a retired
/// identity whose own site a published cell still names.
enum class tet_identity_account : std::uint8_t {
  live,
  retired,
  vanished,
  emitted_dead
};

/// Per identity of `authority` its account against `named`, the site slots
/// the published cells hold, each as often as it likes. O(named + the
/// identities + the sites).
template <typename Index, typename Int, typename Named>
auto account_tet_identities(const tet_identity_authority<Index, Int> &authority,
                            const Named &named, Index n_sites)
    -> tf::buffer<tet_identity_account> {
  tf::buffer<char> held;
  held.allocate_and_initialize(std::size_t(n_sites), char(0));
  for (const auto slot : named)
    held[std::size_t(slot)] = 1;
  tf::buffer<tet_identity_account> accounts;
  accounts.allocate(std::size_t(authority.n_identities()));
  tf::parallel_for_each(
      tf::make_sequence_range(accounts.size()),
      [&authority, &held, &accounts](std::size_t at) {
        const Index id = Index(at);
        const Index seat = authority.seat_of(id);
        const bool shown = seat != Index(-1) && held[std::size_t(seat)] &&
                           authority.identity_at(seat) == id;
        accounts[at] = authority.target(id) != Index(-1)
                           ? (shown ? tet_identity_account::emitted_dead
                                    : tet_identity_account::retired)
                           : (shown ? tet_identity_account::live
                                    : tet_identity_account::vanished);
      },
      tf::checked);
  return accounts;
}

/// Every identity of `authority` accounted against `named`: no retired
/// identity is still shown. A verification check; its caller states that.
template <typename Index, typename Int, typename Named>
auto assert_tet_identities_are_accounted(
    const tet_identity_authority<Index, Int> &authority, const Named &named,
    Index n_sites) -> void {
  const auto accounts = account_tet_identities(authority, named, n_sites);
  for (const auto account : accounts)
    tf::core::verify(account != tet_identity_account::emitted_dead,
                     "a retired identity is named by no published cell");
}

} // namespace tf::tetmesh
