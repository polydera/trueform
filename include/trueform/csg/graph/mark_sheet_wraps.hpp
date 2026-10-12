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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./sheet_sides.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Mark every domain that wraps a sheet's rim — the flush witness of
///        @ref tf::csg::graph::sheet_sides — on the walls' statements.
///
/// A sheet ends on a fan where it stands on exactly one page; the sector
/// holding that page's continuation
/// (@ref tf::csg::graph::plane_radial_fans::continuations) names the domain.
template <typename Index, typename Labels, typename ExposedOfRow>
auto mark_sheet_wraps(tf::csg::graph::sheet_sides<Index> &sides,
                      const tf::csg::graph::arrangement_descriptor<Index> &desc,
                      const tf::csg::graph::component_crossings<Index> &crossings,
                      const Labels &labels, const ExposedOfRow &exposed_of_row)
    -> void {
  using sides_t = tf::csg::graph::sheet_sides<Index>;
  const auto &fans = desc.fans;
  if (sides.n_sheets() == Index(0) || fans.continuations.size() == 0)
    return;

  auto dirs_view = tf::make_offset_block_range(fans.rows.offsets_buffer(),
                                               tf::make_range(fans.dirs));
  // An occurrence's component: an uncut sheet states it, a cut one names
  // its row; `-1` for a row the exposure did not carry.
  auto component_of = [&](Index row) -> Index {
    if (row < Index(0))
      return Index(-row - 1);
    const Index e = exposed_of_row[std::size_t(row / Index(3))];
    return e == Index(-1) ? Index(-1) : Index(labels[e]);
  };
  // The domain of sector `r` of a fan whose pages start at `first`: the
  // side of page `r` facing its successor.
  auto sector_domain = [&](Index first, Index r) -> Index {
    auto page = fans.rows[std::size_t(first + r)];
    auto page_dirs = dirs_view[std::size_t(first + r)];
    for (std::size_t a = 0; a < page.size(); ++a) {
      const Index c = component_of(page[a]);
      if (c >= Index(0))
        return desc.domain_of_side[2 * c + (page_dirs[a] ? 0 : 1)];
    }
    return Index(-1);
  };

  // Per sheet standing on the fan: the pages it stands on, and the last.
  struct standing_t {
    Index sheet;
    Index pages;
    Index page;
  };
  tf::buffer<std::array<Index, 2>> wraps; // (sheet, domain)
  tf::generic_generate(
      tf::make_sequence_range(Index(fans.pieces.size())), wraps,
      [&](Index fan, tf::buffer<std::array<Index, 2>> &out,
          tf::small_vector<standing_t, 4> &standing) {
        if (!desc.valid[std::size_t(fan)])
          return;
        const Index first = fans.page_offsets[std::size_t(fan)];
        const Index K = fans.page_offsets[std::size_t(fan) + 1] - first;
        standing.clear();
        for (Index p = Index(0); p < K; ++p)
          for (const Index row : fans.rows[std::size_t(first + p)]) {
            const Index c = component_of(row);
            if (c < Index(0))
              continue;
            sides.for_each_sheet_layer(desc, crossings, c, [&](Index k, Index) {
              for (auto &s : standing)
                if (s.sheet == k) {
                  ++s.pages;
                  s.page = p;
                  return;
                }
              standing.push_back({k, Index(1), p});
            });
            break;
          }
        for (const auto &s : standing) {
          if (s.pages != Index(1))
            continue;
          const Index r = fans.continuations[std::size_t(first + s.page)];
          if (r < Index(0))
            continue;
          const Index d = sector_domain(first, r);
          if (d >= Index(0))
            out.push_back({s.sheet, d});
        }
      },
      tf::small_vector<standing_t, 4>{});

  tbb::parallel_sort(wraps.begin(), wraps.end());
  wraps.erase_till_end(std::unique(wraps.begin(), wraps.end()));
  for (const auto &w : wraps)
    sides.state(w[0], w[1]) = sides_t::wraps;
}

} // namespace tf::csg::graph
