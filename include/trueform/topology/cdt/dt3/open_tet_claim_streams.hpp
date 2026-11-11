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
#include "./tet_claim_stream.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Open the streams `workers` run to insert `count` sites into a complex
/// holding `sites`, each stream holding no row, with a released claim word
/// beside every row. The lanes reserve the rows a build reserves per site
/// and touch none of them: every band grows the pools by its measured
/// density (@ref grow_tet_claim_pools), within the reserve unless the
/// complex holds more.
template <typename Owner, typename Index>
auto open_tet_claim_streams(Owner &owner, tet_claim_workspace<Index> &work,
                            std::size_t workers, std::size_t sites,
                            std::size_t count) -> void {
  using Tuning = tet_execution_tuning;
  const std::size_t live = owner._corners.size();
  const std::size_t rows =
      live +
      std::min((std::size_t(Owner::max_tets) - live) / Tuning::reserve_per_site,
               count) *
          Tuning::reserve_per_site;
  owner._corners.reserve(rows);
  owner._neighbors.reserve(rows);
  owner._states.reserve(rows);
  work.free_next.reserve(rows);
  work.ownership.reserve(rows);
  const auto streams = std::min(workers, std::size_t(tet_worker_mask) /
                                             Tuning::streams_per_worker) *
                       Tuning::streams_per_worker;
  work.ownership.reallocate_and_initialize(live, tet_unowned);
  work.streams.allocate_and_initialize(streams, tet_claim_stream<Index>{});
  if (work.killed.size() < streams)
    work.killed.resize(streams);
  if (work.links.size() < streams)
    work.links.resize(streams);
  if (work.scratch.size() < streams)
    work.scratch.resize(streams);
  work.inserted = sites;
}

} // namespace tf::topology::cdt::dt3
