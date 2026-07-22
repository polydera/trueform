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
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../cut/arrangements/component_labels.hpp"
#include "../../cut/region_triangulator.hpp"
#include "../../cut/partition/make_ids.hpp"
#include "../../cut/partition/partition_ids.hpp"
#include "../../cut/partition/partition_labels.hpp"
#include "tbb/task_group.h"
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Build per-form selection lists for a CSG output mesh from
///        a precomputed per-component `chosen_sides` array.
///
/// Each form gets a @ref tf::cut::partition_ids with **two labels**:
///   - label `0` → "reverse direction" — components whose
///     `chosen_sides == 0`; emit faces with reversed winding so the
///     result-mesh normals face outward.
///   - label `1` → "forward direction" — components whose
///     `chosen_sides == 1`; emit faces with stored winding.
///
/// Components with `chosen_sides == -1` are skipped (Lévy boundary
/// rule).
///
/// Output is the same shape that
/// @ref tf::cut::make_partition_map_data consumes
/// (`small_vector<partition_ids<Index>, 4>`). Per-form labels are
/// produced in parallel; the counting-sort inside
/// `make_partition_ids` is also parallel internally.
template <typename Index, typename Index1, typename FormsRange>
auto make_csg_partition(const tf::cut::component_labels<Index> &ag,
                        const tf::cut::region_triangulator<Index, Index1> &rt,
                        const FormsRange &forms,
                        const tf::buffer<std::int8_t> &chosen_sides)
    -> tf::small_vector<tf::cut::partition_ids<Index>, 4> {
  using ag_t = tf::cut::component_labels<Index>;
  const Index n_tags = static_cast<Index>(forms.size());

  tf::small_vector<tf::cut::partition_ids<Index>, 4> result;
  result.resize(n_tags);

  auto loop_labels_all = ag.triangle_labels();
  auto tag_offs = rt.tag_offsets();

  tbb::task_group tg;
  for (Index t = 0; t < n_tags; ++t) {
    tg.run([&, t] {
      tf::cut::partition_labels<std::int8_t> labels;
      labels.n_components = std::int8_t(2); // labels: 0=rev, 1=fwd

      // Uncut surface faces → polygon labels.
      auto poly_labels = ag.polygon_labels(t);
      labels.polygon_labels.allocate(poly_labels.size());
      tf::parallel_for_each(
          tf::enumerate(poly_labels),
          [&](auto pair) {
            auto &&[f, c] = pair;
            labels.polygon_labels[f] =
                (c == ag_t::none_label) ? std::int8_t(-1) : chosen_sides[c];
          },
          tf::checked);

      // Cut loops in this form's slice of rt.loops() → cut labels.
      const Index loop_off = tag_offs[t];
      const Index n_loops_t = tag_offs[t + 1] - loop_off;
      labels.cut_labels.allocate(static_cast<std::size_t>(n_loops_t));
      tf::parallel_for_each(
          tf::make_sequence_range(n_loops_t),
          [&](Index i) {
            const Index c = loop_labels_all[loop_off + i];
            labels.cut_labels[i] =
                (c == ag_t::none_label) ? std::int8_t(-1) : chosen_sides[c];
          },
          tf::checked);
      // Promoted faces: conforming triangles carry the face's surface
      // label; the plain face is never selected (refined-path rule).
      for (auto &&[pi, pd] : tf::enumerate(rt.promoted_descriptors())) {
        if (pd.tag != t)
          continue;
        const Index c = poly_labels[pd.object];
        labels.polygon_labels[pd.object] = std::int8_t(-1);
        if (c == ag_t::none_label)
          continue;
        const auto pr = rt.promoted_range(Index(pi));
        for (Index li = pr[0]; li < pr[1]; ++li)
          labels.cut_labels[li - loop_off] = chosen_sides[c];
      }

      result[t] = tf::cut::make_partition_ids<Index>(labels);
    });
  }
  tg.wait();

  return result;
}

} // namespace tf::csg::graph
