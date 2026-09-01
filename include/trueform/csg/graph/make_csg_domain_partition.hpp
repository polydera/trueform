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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "./compute_domain_partition.hpp"
#include "./triangle_component_labels.hpp"
#include "tbb/task_group.h"
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @cond INTERNAL

/// Concatenate the label blocks of two `partition_ids` (each with
/// `n_kept` labels) into one with `2 * n_kept` labels: forward labels
/// occupy `[0, n_kept)`, reverse labels `[n_kept, 2 * n_kept)`.
template <typename Index>
auto concat_partition_ids(const tf::arrangement::partition_ids<Index> &fwd,
                          const tf::arrangement::partition_ids<Index> &rev,
                          Index n_kept)
    -> tf::arrangement::partition_ids<Index> {
  auto concat = [&](const tf::offset_block_buffer<Index, Index> &a,
                    const tf::offset_block_buffer<Index, Index> &b,
                    tf::offset_block_buffer<Index, Index> &out) {
    const auto &a_off = a.offsets_buffer();
    const auto &b_off = b.offsets_buffer();
    auto &off = out.offsets_buffer();
    off.allocate(2 * static_cast<std::size_t>(n_kept) + 1);
    const Index a_total = a_off.size() ? a_off.back() : Index(0);
    for (Index k = 0; k <= n_kept; ++k)
      off[k] = a_off.size() ? a_off[k] : Index(0);
    for (Index k = 1; k <= n_kept; ++k)
      off[n_kept + k] = a_total + (b_off.size() ? b_off[k] : Index(0));

    auto &data = out.data_buffer();
    data.allocate(a.data_buffer().size() + b.data_buffer().size());
    tf::parallel_copy(tf::make_range(a.data_buffer()),
                      tf::slice(data, 0, a.data_buffer().size()));
    tf::parallel_copy(tf::make_range(b.data_buffer()),
                      tf::drop(data, a.data_buffer().size()));
  };
  tf::arrangement::partition_ids<Index> out;
  concat(fwd.polygons, rev.polygons, out.polygons);
  concat(fwd.cut_faces, rev.cut_faces, out.cut_faces);
  return out;
}

/// @endcond

/// @ingroup csg_graph_internals
/// @brief Build per-form selection lists keyed on the kept-domain a
///        component-side bounds, for per-domain CSG extraction.
///
/// Each form gets a @ref tf::arrangement::partition_ids with `2 * n_kept`
/// labels. For output domain `k` in `[0, n_kept)`:
///   - label `k` → that form's faces/loops whose **forward** side
///     (`side_label[2c + 1] == k`) bounds domain `k`; emit with stored
///     winding.
///   - label `n_kept + k` → faces/loops whose **reverse** side
///     (`side_label[2c + 0] == k`) bounds domain `k`; emit reversed so
///     the cell's normals point outward.
///
/// A component with two distinct kept sides therefore lands in two
/// different domains' label blocks. Sentinel-bound sides
/// (`side_label == -1`, i.e. self-merged flaps or non-kept domains) are
/// dropped.
template <typename Index, typename Arrangement, typename TagMask = tf::none_t>
auto make_csg_domain_partition(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const domain_partition<Index> &part, const TagMask &tag_mask = {})
    -> tf::small_vector<tf::arrangement::partition_ids<Index>, 4> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;
  const Index n_tags = arrangement.n_tags();
  const Index n_kept = part.n_kept;
  const auto &side_label = part.side_label;

  tf::small_vector<tf::arrangement::partition_ids<Index>, 4> result;
  result.resize(n_tags);

  auto triangle_labels = labels.triangle_labels();
  auto tag_offsets = arrangement.global().tag_offsets();

  auto side_to_label = [&](Index c, Index s) -> Index {
    return side_label[2 * c + s]; // dense kept id in [0, n_kept), or -1
  };

  tbb::task_group tg;
  for (Index t = 0; t < n_tags; ++t) {
    tg.run([&, t] {
      // A masked-out tag contributes nothing: its faces still cut (they
      // are in the arrangement), they just do not emit — every cell
      // opens along the walls that form would have contributed.
      if constexpr (!std::is_same_v<TagMask, tf::none_t>)
        if (!tag_mask[std::size_t(t)]) {
          tf::arrangement::partition_labels<Index> empty;
          empty.n_components = 2 * n_kept;
          result[t] = tf::arrangement::make_partition_ids<Index>(empty);
          return;
        }
      auto poly_labels = labels.polygon_labels(t);
      const Index tri_off = tag_offsets[t];
      const Index n_tris_t = tag_offsets[t + 1] - tri_off;

      auto build_side = [&](Index s) {
        tf::arrangement::partition_labels<Index> lab;
        lab.n_components = n_kept;
        lab.polygon_labels.allocate(poly_labels.size());
        tf::parallel_for_each(
            tf::enumerate(poly_labels),
            [&](auto pair) {
              auto &&[f, c] = pair;
              lab.polygon_labels[f] =
                  (c == labels_t::none_label) ? Index(-1)
                                              : side_to_label(c, s);
            },
            tf::checked);
        lab.cut_labels.allocate(static_cast<std::size_t>(n_tris_t));
        tf::parallel_for_each(
            tf::make_sequence_range(n_tris_t),
            [&](Index i) {
              const Index c = triangle_labels[tri_off + i];
              lab.cut_labels[i] =
                  (c == labels_t::none_label) ? Index(-1)
                                              : side_to_label(c, s);
            },
            tf::checked);
        return tf::arrangement::make_partition_ids<Index>(lab);
      };

      auto fwd_ids = build_side(Index(1));
      auto rev_ids = build_side(Index(0));
      result[t] = concat_partition_ids(fwd_ids, rev_ids, n_kept);
    });
  }
  tg.wait();

  return result;
}

} // namespace tf::csg::graph
