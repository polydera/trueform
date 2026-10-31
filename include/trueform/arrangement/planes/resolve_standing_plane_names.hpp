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
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// THE NAME OUTLIVES THE WAVE. Equal names are one point by construction,
/// and the plane layer keeps the name of every class a wave minted, so a
/// fresh class whose name a standing class carries speaks the identity that
/// class minted — what it speaks now, through the closed plane merges —
/// and never mints a coordinate twin of it. A landing name speaks its point
/// already and is not looked up. `class_id` holds `-1` for every class the
/// wave has not yet named; the standing names are indexed once per wave.
template <typename Index, typename Param>
auto resolve_standing_plane_names(
    const tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    const tf::buffer<Index> &class_offsets, Index n_classes,
    const tf::buffer<plane_recovery_name<Index, Param>> &class_names,
    const tf::buffer<Index> &created_classes,
    const tf::buffer<std::array<Index, 3>> &merges, Index base_created,
    tf::buffer<Index> &class_id) -> void {
  using name_t = plane_recovery_name<Index, Param>;
  struct row_t {
    name_t name;
    Index cls;
  };
  if (n_classes == Index(0) || class_names.size() == 0)
    return;
  tf::buffer<row_t> rows;
  rows.allocate(class_names.size());
  tf::parallel_for_each(
      tf::make_sequence_range(class_names.size()),
      [&](std::size_t cls) {
        rows[cls] = {class_names[cls], Index(cls)};
      },
      tf::checked);
  tbb::parallel_sort(rows.begin(), rows.end(),
                     [](const row_t &x, const row_t &y) {
                       return x.name < y.name;
                     });
  // the identity each class minted: one per class, so the inverse is a
  // table written once at distinct slots; a direct refined point names no
  // class and is skipped
  tf::buffer<Index> minted;
  minted.allocate_and_initialize(class_names.size(), Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(created_classes.size()),
      [&](std::size_t created) {
        const auto cls = created_classes[created];
        if (cls >= Index(0))
          minted[std::size_t(cls)] = Index(created);
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_classes)),
      [&](std::size_t cls) {
        if (class_id[cls] != Index(-1))
          return;
        const auto &name = statements[std::size_t(class_offsets[cls])].name;
        const auto at = std::lower_bound(
            rows.begin(), rows.end(), name,
            [](const row_t &row, const name_t &key) { return row.name < key; });
        if (at == rows.end() || !(at->name == name))
          return;
        const auto created = minted[std::size_t(at->cls)];
        if (created == Index(-1))
          return;
        const auto id = base_created + created;
        const auto target =
            tf::intersect::graph::plane_merge_target(merges, Index(1), id);
        class_id[cls] = target == Index(-1) ? id : target;
      },
      tf::checked);
}

} // namespace tf::arrangement
