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
#include "../../../core/views/mapped_range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./measure_tet_site_domain.hpp"
#include "./tet_position.hpp"
#include "./tet_site_morton_key.hpp"
#include "./tet_site_position.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Order the `count` appended sites, the last of `owner`'s, for insertion:
/// the Morton order of their floors over their own domain, equal keys by
/// position then name.
template <typename Owner>
auto order_appended_tet_sites(Owner &owner, std::size_t count) -> void {
  using Index = typename Owner::index_type;
  if (count < 2)
    return;
  const std::size_t first = tet_site_count(owner) - count;
  const auto appended = tf::make_mapped_range(
      tf::make_sequence_range(first, first + count),
      [&owner](std::size_t site) {
        return tet_site_position(owner, Index(site)).floor;
      });
  const auto domain =
      measure_tet_site_domain<false, typename Owner::coord_type>(appended);
  const auto shift = tet_site_morton_shift(domain);
  auto &keyed = owner._appended_keys;
  keyed.allocate(count);
  for (std::size_t i = 0; i < count; ++i)
    keyed[i] = {tet_site_morton_key(appended[i], domain, shift),
                Index(first + i)};
  std::sort(keyed.begin(), keyed.end(),
            [&owner](const std::pair<std::uint64_t, Index> &x,
                     const std::pair<std::uint64_t, Index> &y) {
              if (x.first != y.first)
                return x.first < y.first;
              const auto a = tet_site_position(owner, x.second);
              const auto b = tet_site_position(owner, y.second);
              return a < b || (a == b && tet_site_name(owner, x.second) <
                                             tet_site_name(owner, y.second));
            });
  for (std::size_t i = 0; i < count; ++i)
    owner._order[i] = keyed[i].second;
}

} // namespace tf::topology::cdt::dt3
