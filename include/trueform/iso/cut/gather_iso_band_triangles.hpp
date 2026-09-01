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

#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slide_range.hpp"
#include "../../core/views/take.hpp"

#include <cstddef>

namespace tf::iso {

/// CORE. The kept pieces' triangles in band-major order. Every piece's block
/// length is known before a triangle moves, so counts and ONE prefix give each
/// piece an exact disjoint range and the stream is written straight into it.
///
/// `band_regions` is a range of per-band region-id blocks. `band_offsets`
/// comes back in TRIANGLE units, which is what a per-face label range reads.
template <typename Index, typename BandRegions, typename Regions>
auto gather_iso_band_triangles(const BandRegions &band_regions,
                               const Regions &regions,
                               tf::blocked_buffer<Index, 3> &triangles,
                               tf::buffer<Index> &origins,
                               tf::buffer<Index> &band_offsets) -> void {
  const auto n_bands = std::size_t(band_regions.size());
  tf::buffer<Index> band_region_offsets;
  band_region_offsets.allocate(n_bands + 1);
  band_region_offsets[0] = 0;
  {
    std::size_t band = 0;
    for (const auto block : band_regions) {
      band_region_offsets[band + 1] =
          band_region_offsets[band] + static_cast<Index>(block.size());
      ++band;
    }
  }
  const auto n_pieces = std::size_t(band_region_offsets[n_bands]);

  tf::buffer<Index> pieces;
  pieces.allocate(n_pieces);
  {
    std::size_t band = 0;
    for (const auto block : band_regions) {
      tf::parallel_copy(
          block,
          tf::take(tf::drop(pieces,
                            std::size_t(band_region_offsets[band])),
                   block.size()));
      ++band;
    }
  }

  tf::buffer<Index> starts;
  starts.allocate(n_pieces + 1);
  starts[0] = 0;
  tf::parallel_for_each(
      tf::enumerate(tf::make_range(pieces)),
      [&](auto pair) {
        auto [piece, region] = pair;
        starts[std::size_t(piece) + 1] = static_cast<Index>(
            regions.triangles[std::size_t(region)].size());
      },
      tf::checked);
  for (auto pair : tf::make_slide_range<2>(starts))
    pair[1] += pair[0];

  band_offsets.allocate(n_bands + 1);
  tf::parallel_for_each(
      tf::make_sequence_range(n_bands + 1),
      [&](std::size_t band) {
        band_offsets[band] = starts[std::size_t(band_region_offsets[band])];
      },
      tf::checked);

  const auto n_triangles = std::size_t(starts[n_pieces]);
  triangles.allocate(n_triangles);
  origins.allocate(n_triangles);
  tf::parallel_for_each(
      tf::make_sequence_range(n_pieces),
      [&](std::size_t piece) {
        const auto region = std::size_t(pieces[piece]);
        const auto object = regions.faces[region];
        auto slot = std::size_t(starts[piece]);
        for (const auto &triangle : regions.triangles[region]) {
          auto corners = triangles[slot];
          corners[0] = triangle[0];
          corners[1] = triangle[1];
          corners[2] = triangle[2];
          origins[slot] = object;
          ++slot;
        }
      },
      tf::checked);
}

} // namespace tf::iso
