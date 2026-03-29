/*
 * Copyright (c) 2025 XLAB
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

#include "../../core/algorithm/mask_to_index_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/segments_buffer.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/exact/segment_intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"

namespace tf::cut {

/// Construct split segments + edge labels from a segment intersection graph.
///
/// Output segments have points:
///   [graph intersection points (deconverted) | original segment points]
///
/// Edge labels: for each output edge, the index of the original edge.
template <typename Index, typename RealType, std::size_t Dims, typename Policy>
auto make_segment_arrangements(
    const tf::intersect::segment_intersection_graph<Index, RealType, Dims> &sig,
    const tf::segments<Policy> &segments,
    const tf::exact::pt_converter<RealType, Dims> &conv)
    -> std::pair<tf::segments_buffer<Index, RealType, Dims>,
                 tf::buffer<Index>> {

  tf::buffer<bool> original_e_mask;
  original_e_mask.allocate(segments.edges().size());
  tf::parallel_fill(original_e_mask, true);
  tf::parallel_fill(
      tf::make_indirect_range(sig.origin_edges(), original_e_mask), false);

  auto e_im = tf::mask_to_index_map<Index>(original_e_mask);

  tf::buffer<bool> original_v_mask;
  original_v_mask.allocate(segments.points().size());
  tf::parallel_fill(original_v_mask, false);
  tf::parallel_for_each(sig.flat_sub_edges(), [&](auto edge) {
    if (edge[0].source == tf::intersect::graph::vertex_source::original)
      original_v_mask[edge[0].id] = true;
    if (edge[1].source == tf::intersect::graph::vertex_source::original)
      original_v_mask[edge[1].id] = true;
  });
  tf::parallel_for_each(
      tf::make_indirect_range(e_im.kept_ids(), segments.edges()),
      [&](auto edge) {
        original_v_mask[edge[0]] = true;
        original_v_mask[edge[1]] = true;
      });

  auto v_im = tf::mask_to_index_map<Index>(original_v_mask);

  auto map_v = [&, n = v_im.kept_ids().size()](auto v) {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return Index(v_im.f()[v.id]);
    else
      return Index(v.id + n);
  };

  auto mapped_edges = tf::make_indirect_range(
      e_im.kept_ids(), tf::make_mapped_range(segments.edges(), [&](auto edge) {
        return std::array<Index, 2>{Index(v_im.f()[edge[0]]),
                                    Index(v_im.f()[edge[1]])};
      }));

  auto mapped_new_edges =
      tf::make_mapped_range(sig.flat_sub_edges(), [&](auto edge) {
        return std::array<Index, 2>{map_v(edge[0]), map_v(edge[1])};
      });

  auto total_pts = v_im.kept_ids().size() + sig.points().size();
  auto total_e = e_im.kept_ids().size() + sig.flat_sub_edges().size();

  // Build output points: [deconverted graph pts | original segment pts]
  tf::segments_buffer<Index, RealType, Dims> out;
  out.points_buffer().allocate(total_pts);
  tf::parallel_copy(tf::make_indirect_range(v_im.kept_ids(), segments.points()),
                    tf::take(out.points(), v_im.kept_ids().size()));

  tf::parallel_copy(
      tf::make_mapped_range(sig.points(),
                            [&](auto pt) { return conv.deconvert(pt); }),
      tf::drop(out.points(), v_im.kept_ids().size()));

  out.edges_buffer().allocate(total_e);
  tf::parallel_copy(mapped_edges, tf::take(out.edges(), mapped_edges.size()));
  tf::parallel_copy(mapped_new_edges,
                    tf::drop(out.edges(), mapped_edges.size()));

  tf::buffer<Index> origins;
  origins.allocate(total_e);
  tf::parallel_copy(e_im.kept_ids(), tf::take(origins, e_im.kept_ids().size()));

  auto o_blocks =
      tf::make_offset_block_range(sig.sub_edges_buffer().offsets_buffer(),
                                  tf::drop(origins, e_im.kept_ids().size()));

  tf::parallel_for_each(tf::zip(sig.origin_edges(), o_blocks), [](auto pair) {
      auto &&[origin, edge] = pair;
    for (auto &e : edge)
      e = origin;
  });

  return {std::move(out), std::move(origins)};
}

} // namespace tf::cut
