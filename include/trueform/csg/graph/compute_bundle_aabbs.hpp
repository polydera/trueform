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
#include "../../core/aabb.hpp"
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../cut/arrangement_graph.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include "../../cut/face_cuts.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"
#include <limits>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Exact per-bundle AABB in SoS coordinates.
///
/// Walks every original face vertex whose owning component is in a
/// bundle, plus every cut-loop vertex, aggregating min/max into the
/// returned `bundle_aabb` per bundle. Nested parallel reduce: outer
/// over tags, inner over each tag's faces; loops are a single flat
/// reduce. The inner reduce writes into the outer task's local
/// accumulator, so sibling outer tasks never collide.
template <typename Forms, typename Index, typename Int, typename Real,
          std::size_t Dims>
auto compute_bundle_aabbs(
    const tf::cut::arrangement_descriptor<Index> &desc,
    const tf::arrangement_graph<Index> &ag,
    const tf::face_cuts<Index, Int> &fc,
    const tf::intersection_graph<Index, Int> &ig,
    const Forms &tagged_forms,
    const tf::exact::vertex_converter<Int, Real, Dims> &conv)
    -> tf::buffer<tf::aabb<Int, 3>> {
  using ag_t = tf::arrangement_graph<Index>;
  using EVert = tf::exact::vertex<Index, Int>;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;
  using bbox_t = tf::aabb<Int, 3>;

  const Index n_bundles = desc.n_bundles;
  const Index n_tags = ag.n_tags();
  const Int int_max = std::numeric_limits<Int>::max();
  const Int int_min = std::numeric_limits<Int>::min();

  bbox_t init_bbox = tf::make_aabb(tf::make_point(int_max, int_max, int_max),
                                   tf::make_point(int_min, int_min, int_min));
  tf::buffer<bbox_t> bboxes;
  bboxes.allocate(static_cast<std::size_t>(n_bundles));
  for (Index b = Index(0); b < n_bundles; ++b)
    bboxes[b] = init_bbox;

  if (n_tags == Index(0))
    return bboxes;

  const Index n_orig_total =
      Index(conv.offsets[static_cast<int>(n_tags - 1)]) +
      Index(tagged_forms[n_tags - 1].points().size());

  auto get_original_vertex = [&](Index idx, Index tag) -> EVert {
    auto world = tf::transformed(tagged_forms[tag].points()[idx],
                                 tf::frame_of(tagged_forms[tag]));
    return {Index(idx) + Index(conv.offsets[static_cast<int>(tag)]),
            conv.convert(world)};
  };

  auto get_vertex = [&](const vertex_t &v, Index tag) -> EVert {
    if (v.source == source::created)
      return {n_orig_total + Index(v.id), ig.points()[v.id]};
    return get_original_vertex(Index(v.id), tag);
  };

  tf::buffer<bbox_t> bbox_prototype;
  bbox_prototype.allocate(static_cast<std::size_t>(n_bundles));
  tf::parallel_fill(bbox_prototype, init_bbox);

  auto combine_bboxes = [](const tf::buffer<bbox_t> &local,
                           tf::buffer<bbox_t> &out) {
    for (decltype(out.size()) b = 0; b < out.size(); ++b) {
      for (int k = 0; k < 3; ++k) {
        if (local[b].min[k] < out[b].min[k])
          out[b].min[k] = local[b].min[k];
        if (local[b].max[k] > out[b].max[k])
          out[b].max[k] = local[b].max[k];
      }
    }
  };

  tf::blocked_reduce(
      tf::make_sequence_range(n_tags), bboxes, bbox_prototype,
      [&](auto &&tag_block, tf::buffer<bbox_t> &outer_local) {
        for (Index t : tag_block) {
          auto labels = ag.polygon_labels(t);
          auto faces = tagged_forms[t].faces();
          if (faces.size() == 0)
            continue;
          tf::blocked_reduce(
              tf::enumerate(faces), outer_local, bbox_prototype,
              [&, t](auto &&block, tf::buffer<bbox_t> &local) {
                for (const auto &pair : block) {
                  const auto &[f, face] = pair;
                  const Index c = labels[Index(f)];
                  if (c == ag_t::none_label)
                    continue;
                  const Index b = desc.bundle_of_component[c];
                  const auto n_fv = face.size();
                  for (std::size_t i = 0; i < n_fv; ++i) {
                    auto ev = get_original_vertex(Index(face[i]), t);
                    for (int k = 0; k < 3; ++k) {
                      if (ev.pt[k] < local[b].min[k])
                        local[b].min[k] = ev.pt[k];
                      if (ev.pt[k] > local[b].max[k])
                        local[b].max[k] = ev.pt[k];
                    }
                  }
                }
              },
              combine_bboxes);
        }
      },
      combine_bboxes);

  auto loops = fc.loops();
  auto descs = fc.descriptors();
  auto loop_labels = ag.loop_labels();
  if (loops.size() == 0)
    return bboxes;

  tf::blocked_reduce(
      tf::enumerate(loops), bboxes, bbox_prototype,
      [&](auto &&block, tf::buffer<bbox_t> &local) {
        for (const auto &pair : block) {
          const auto &[l, loop] = pair;
          const Index c = loop_labels[Index(l)];
          if (c == ag_t::none_label)
            continue;
          const Index tag = descs[Index(l)].tag;
          if (tag == Index(-1))
            continue;
          const Index b = desc.bundle_of_component[c];
          const auto n_lv = loop.size();
          for (std::size_t i = 0; i < n_lv; ++i) {
            auto ev = get_vertex(loop[i], tag);
            for (int k = 0; k < 3; ++k) {
              if (ev.pt[k] < local[b].min[k])
                local[b].min[k] = ev.pt[k];
              if (ev.pt[k] > local[b].max[k])
                local[b].max[k] = ev.pt[k];
            }
          }
        }
      },
      combine_bboxes);

  return bboxes;
}

} // namespace tf::csg::graph
