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
#include "../../arrangement/partition/make_ids.hpp"
#include "../../arrangement/partition/partition_ids.hpp"
#include "../../arrangement/partition/partition_labels.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./triangle_component_labels.hpp"
#include "tbb/task_group.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Build per-form selection lists for a CSG output mesh from
///        a precomputed per-component `chosen_sides` array.
///
/// Each form gets a @ref tf::arrangement::partition_ids with **two labels**:
///   - label `0` → "reverse direction" — components whose
///     `chosen_sides == 0`; emit faces with reversed winding so the
///     result-mesh normals face outward.
///   - label `1` → "forward direction" — components whose
///     `chosen_sides == 1`; emit faces with stored winding.
///
/// Components with `chosen_sides == -1` are skipped, as are the carriers
/// the label tier left unlabelled: a cut face (its triangles carry the
/// labels instead) and a dead coplanar duplicate.
///
/// Output is the shape @ref tf::csg::graph::make_csg_map_data consumes
/// (`small_vector<partition_ids<Index>, 4>`). Per-form labels are
/// produced in parallel; the counting-sort inside
/// `make_partition_ids` is also parallel internally.
template <typename Index, typename Arrangement, typename TagMask = tf::none_t>
auto make_csg_partition(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const tf::buffer<std::int8_t> &chosen_sides, const TagMask &tag_mask = {})
    -> tf::small_vector<tf::arrangement::partition_ids<Index>, 4> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;

  const Index n_tags = arrangement.n_tags();
  tf::small_vector<tf::arrangement::partition_ids<Index>, 4> result;
  result.resize(n_tags);

  auto triangle_labels = labels.triangle_labels();
  auto tag_offsets = arrangement.global().tag_offsets();

  tbb::task_group tg;
  for (Index t = 0; t < n_tags; ++t) {
    tg.run([&, t] {
      // A masked-out tag contributes nothing: its faces still CUT the
      // others (they are in the arrangement), they just do not emit.
      if constexpr (!std::is_same_v<TagMask, tf::none_t>)
        if (!tag_mask[std::size_t(t)]) {
          tf::arrangement::partition_labels<std::int8_t> empty;
          empty.n_components = std::int8_t(2);
          result[t] = tf::arrangement::make_partition_ids<Index>(empty);
          return;
        }
      tf::arrangement::partition_labels<std::int8_t> sides;
      sides.n_components = std::int8_t(2); // labels: 0=rev, 1=fwd

      // Uncut surface faces → polygon labels.
      auto polygon_labels = labels.polygon_labels(t);
      sides.polygon_labels.allocate(polygon_labels.size());
      tf::parallel_for_each(
          tf::enumerate(polygon_labels),
          [&](auto pair) {
            auto &&[f, c] = pair;
            sides.polygon_labels[f] =
                (c == labels_t::none_label) ? std::int8_t(-1) : chosen_sides[c];
          },
          tf::checked);

      // This form's slice of the exposed triangle stream → cut labels.
      const Index lo = tag_offsets[t];
      const Index n_tris = tag_offsets[t + 1] - lo;
      sides.cut_labels.allocate(static_cast<std::size_t>(n_tris));
      tf::parallel_for_each(
          tf::make_sequence_range(n_tris),
          [&](Index i) {
            const Index c = triangle_labels[lo + i];
            sides.cut_labels[i] =
                (c == labels_t::none_label) ? std::int8_t(-1) : chosen_sides[c];
          },
          tf::checked);

      result[t] = tf::arrangement::make_partition_ids<Index>(sides);
    });
  }
  tg.wait();

  return result;
}

} // namespace tf::csg::graph
