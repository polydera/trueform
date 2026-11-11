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
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::tetmesh {

/// How a drain ends the drains, or `none` where they go on: `converged`, no
/// crossing named; `unretired`, a drain that retired no facet; and the
/// refusals — `failed`, a wave that left a carrier refusing or a T-junction;
/// `budget`, mints the site budget cannot hold; `withdrawn`, a mint the
/// complex withdrew; `stalled`, a drain that neither minted nor reduced the
/// crossings it was handed.
enum class tet_drain_end : std::uint8_t {
  none,
  converged,
  unretired,
  failed,
  budget,
  withdrawn,
  stalled
};

/// THE DRAINS' MEASURE: every drain either MINTS, consuming the one site
/// budget its mints share with the recovery's splits and face points, or
/// strictly REDUCES the crossings it was handed; a drain that does neither
/// is `stalled`. The pair (sites the budget has left, crossings handed)
/// falls lexicographically with every drain that goes on, so the drains are
/// finite: at most `budget` minting drains, and between two of them no more
/// drains than the crossings handed to the first. `budget` is the sites the
/// complex may hold.
struct tet_drain_measure {
  std::size_t budget = 0;
  std::size_t handed = std::numeric_limits<std::size_t>::max();
  bool minted = true;

  /// Before a drain handed `crossings`.
  auto open(std::size_t crossings) -> tet_drain_end {
    if (!crossings)
      return tet_drain_end::converged;
    if (!minted && crossings >= handed)
      return tet_drain_end::stalled;
    handed = crossings;
    return tet_drain_end::none;
  }
  /// After its cut: the triangles it failed and the facets it retired.
  auto cut(std::size_t failed, std::size_t retired) const -> tet_drain_end {
    if (failed)
      return tet_drain_end::failed;
    if (!retired)
      return tet_drain_end::unretired;
    return tet_drain_end::none;
  }
  /// Before its mints, `sites` standing and `mints` to place.
  auto mint(std::size_t sites, std::size_t mints) -> tet_drain_end {
    if (sites > budget || mints > budget - sites)
      return tet_drain_end::budget;
    minted = mints != 0;
    return tet_drain_end::none;
  }
  /// After its mints, `withdrawn` of them holding no site.
  auto seat(std::size_t withdrawn) const -> tet_drain_end {
    return withdrawn ? tet_drain_end::withdrawn : tet_drain_end::none;
  }
};

} // namespace tf::tetmesh
