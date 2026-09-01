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
#include "../../core/checked.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./plane_refinement_physical_split.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// State accepted canonical refinement splits on their whole original sides.
///
/// `definitions_of(split)` supplies the split's complete fused definition
/// span. A representative definition is insufficient: every whole-side
/// instance is an independent physical delivery path into the source forms.
template <typename Index, typename Int, typename Splits, typename DefinitionsOf,
          typename World, typename ApplyToForm>
auto make_plane_refinement_physical_splits(
    const Splits &splits, const DefinitionsOf &definitions_of,
    const World &world, const ApplyToForm &apply_to_form,
    tf::buffer<plane_refinement_physical_split<Index, Int>> &physical) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  const param_t maximum = param_t(1) << tf::exact::meta<Int>::param_bits;

  physical.clear();
  tf::generic_generate(
      splits, physical,
      [&](const auto &split,
          tf::buffer<plane_refinement_physical_split<Index, Int>> &out) {
        for (const auto &def : definitions_of(split)) {
          if (!(def.flags & tf::intersect::graph::plane_edge_whole_side_flag))
            continue;
          const auto &descriptor = world.descriptor(def.face);
          apply_to_form(Index(descriptor.tag), [&](const auto &form) {
            const auto face = form.faces()[descriptor.object];
            const auto side = std::size_t(def.side);
            const auto a = Index(face[side]);
            const auto b = Index(face[(side + 1) % face.size()]);
            const auto u = std::min(a, b);
            const auto v = std::max(a, b);
            const bool reversed =
                bool(def.flags &
                     tf::intersect::graph::plane_edge_reversed_flag) !=
                (a != u);
            out.push_back({Index(descriptor.tag), u, v,
                           reversed ? param_t(maximum - split.parameter)
                                    : split.parameter});
          });
        }
      },
      tf::checked);
  if (physical.size() == 0)
    return;
  tbb::parallel_sort(physical.begin(), physical.end());
  physical.erase_till_end(std::unique(physical.begin(), physical.end()));
}

} // namespace tf::arrangement
