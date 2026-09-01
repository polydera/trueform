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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../records/point_delivery.hpp"
#include "../records/tagged_intersection.hpp"
#include "./identify_vertices.hpp"
#include "./identity_records.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect {

/// Give the duplicator's shared-vertex deliveries their point names.
///
/// Identity is settled before the fan, but a self pair's shared
/// vertices are discovered per fan destination, so a delivered record
/// arrives carrying a sentinel — the flat vertex id offset past the
/// fact-slot count. A delivered vertex that some pre-duplication record
/// anchored resolves to its existing kind-V point; the rest are new
/// kind-V points, appended after the existing anchors, which shifts
/// every kind-E id up by their count.
///
/// The renaming is one global fact, so it is stated once over both of
/// the fan's carriers. A sentinel is minted by the self family, which
/// fans whole, so every one of them is a PAIR record and the pair
/// currency is what the scan reads.
template <typename Index, typename Int>
auto resolve_delivered_vertices(
    tf::buffer<tf::intersect::tagged_intersection<Index>> &records,
    tf::buffer<point_delivery<Index>> &deliveries, Index sentinel_base,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<std::array<Index, 2>> &identifications,
    point_identities<Index, Int> &identities) -> void {
  tf::buffer<Index> delivered;
  tf::generic_generate(
      tf::make_range(records), delivered, [&](const auto &rec, auto &buf) {
        if (rec.id >= sentinel_base)
          buf.push_back(canonical_flat_vertex(identifications,
                                              rec.id - sentinel_base));
      },
      tf::checked);
  if (delivered.size() == 0)
    return;
  tbb::parallel_sort(delivered.begin(), delivered.end());
  delivered.erase_till_end(std::unique(delivered.begin(), delivered.end()));

  const auto n_old = Index(identities.vertex_anchors.size());
  auto flat_of = [&](const vertex_anchor<Index> &a) -> Index {
    return vertex_offsets[std::size_t(a.tag)] + a.vid;
  };
  // Anchors were minted in ascending canonical order, so flat ids are
  // binary-searchable.
  auto old_index = [&](Index flat) -> Index {
    auto it = std::lower_bound(
        identities.vertex_anchors.begin(),
        identities.vertex_anchors.begin() + n_old, flat,
        [&](const vertex_anchor<Index> &a, Index f) { return flat_of(a) < f; });
    if (it != identities.vertex_anchors.begin() + n_old && flat_of(*it) == flat)
      return Index(it - identities.vertex_anchors.begin());
    return n_old;
  };

  tf::buffer<Index> fresh;
  for (const auto flat : delivered)
    if (old_index(flat) == n_old)
      fresh.push_back(flat);
  const auto k = Index(fresh.size());
  for (const auto flat : fresh) {
    const auto tag = int(std::upper_bound(vertex_offsets.begin(),
                                          vertex_offsets.end(), flat) -
                         vertex_offsets.begin()) -
                     1;
    identities.vertex_anchors.push_back(
        {std::int16_t(tag), flat - vertex_offsets[std::size_t(tag)]});
  }
  auto fresh_index = [&](Index flat) -> Index {
    return n_old + Index(std::lower_bound(fresh.begin(), fresh.end(), flat) -
                         fresh.begin());
  };

  if (k != 0)
    tf::parallel_for_each(
        tf::make_range(identities.edge_splits.data_buffer()),
        [&](Index &id) {
          if (id >= n_old)
            id += k;
        },
        tf::checked);

  const auto rename = [&](Index &id) {
    if (id >= sentinel_base) {
      const auto flat =
          canonical_flat_vertex(identifications, id - sentinel_base);
      const auto at = old_index(flat);
      id = at != n_old ? at : fresh_index(flat);
    } else if (k != 0 && id >= n_old) {
      id += k;
    }
  };
  tf::parallel_for_each(
      tf::make_range(deliveries),
      [&](point_delivery<Index> &rec) { rename(rec.id); }, tf::checked);
  tf::parallel_for_each(
      tf::make_range(records),
      [&](tf::intersect::tagged_intersection<Index> &rec) { rename(rec.id); },
      tf::checked);
}

} // namespace tf::intersect
