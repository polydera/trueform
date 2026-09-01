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
#include "../../core/algorithm/sparse_block_accumulator.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./arrangement_descriptor.hpp"
#include "./triangle_component_labels.hpp"
#include <cstddef>
#include <limits>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Exact per-bundle AABB in SoS coordinates.
///
/// Two disjoint carriers: each form's uncut faces (their own local
/// point ids) and the arrangement's exposed triangle stream (created
/// ids plus flat original ids). Nested parallel reduce: outer over
/// tags, inner over each tag's faces; the stream is a single flat
/// reduce. The inner reduce writes into the outer task's local
/// accumulator, so sibling outer tasks never collide.
///
/// A single-member carrier holds only its own tag's originals, so
/// `own_tag` names the owning form outright. Only a pooled carrier can
/// hand a triangle another form's corner; there `own_tag` is -1 and the
/// flat id is searched.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename ReadPoint>
auto compute_bundle_aabbs(
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form, const ReadPoint &read_point)
    -> tf::buffer<tf::aabb<Int, 3>> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;
  using source = tf::intersect::graph::vertex_source;
  using bbox_t = tf::aabb<Int, 3>;

  const Index n_bundles = desc.n_bundles;
  const Index n_tags = arrangement.n_tags();
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

  // THE BUNDLE AXIS IS GLOBAL AND A BLOCK TOUCHES A RUN OF IT. A block-local
  // dense array over it makes every block pay for every bundle in the
  // arrangement — and on the branch where a bundle IS a component
  // (@ref tf::csg::graph::make_arrangement_descriptor with no fans) that axis
  // is the whole component space. The block carries the bundles it names; a
  // box is a min/max, so the order they are grown in cannot change one.
  using local_t = tf::sparse_block_accumulator<Index, bbox_t>;
  const auto grow = [](bbox_t &box, const tf::point<Int, 3> &pt) {
    for (int k = 0; k < 3; ++k) {
      if (pt[k] < box.min[k])
        box.min[k] = pt[k];
      if (pt[k] > box.max[k])
        box.max[k] = pt[k];
    }
  };
  const auto merge = [](bbox_t &box, const bbox_t &other) {
    for (int k = 0; k < 3; ++k) {
      if (other.min[k] < box.min[k])
        box.min[k] = other.min[k];
      if (other.max[k] > box.max[k])
        box.max[k] = other.max[k];
    }
  };
  // the inner reduce folds into the OUTER task's own accumulator, so sibling
  // outer tasks never collide; the outer folds into the dense table
  auto combine_local = [merge, init_bbox](const local_t &local,
                                          local_t &outer) {
    for (const auto &entry : local.entries)
      merge(outer.touch(entry.key, init_bbox), entry.value);
  };
  auto combine_bboxes = [merge](const local_t &local,
                                tf::buffer<bbox_t> &out) {
    for (const auto &entry : local.entries)
      merge(out[entry.key], entry.value);
  };

  tf::blocked_reduce(
      tf::make_sequence_range(n_tags), bboxes, local_t{},
      [&](auto &&tag_block, local_t &outer_local) {
        outer_local.clear();
        for (Index t : tag_block)
          apply_to_form(t, [&, t](const auto &form) {
            auto point_of = [&, t](Index idx) {
              return read_point(int(t), idx);
            };
            auto polygon_labels = labels.polygon_labels(t);
            auto faces = form.faces();
            if (faces.size() == 0)
              return;
            tf::blocked_reduce(
                tf::enumerate(faces), outer_local, local_t{},
                [&](auto &&block, local_t &local) {
                  local.clear();
                  for (const auto &pair : block) {
                    const auto &[f, face] = pair;
                    const Index c = polygon_labels[Index(f)];
                    if (c == labels_t::none_label)
                      continue;
                    auto &box =
                        local.touch(desc.bundle_of_component[c], init_bbox);
                    const auto n_fv = face.size();
                    for (std::size_t i = 0; i < n_fv; ++i)
                      grow(box, point_of(Index(face[i])));
                  }
                },
                combine_local);
          });
      },
      combine_bboxes);

  const auto &ga = arrangement.global();
  auto tris = ga.exposed_tris();
  if (tris.size() == 0)
    return bboxes;

  auto triangle_labels = labels.triangle_labels();
  const auto &created = arrangement.created_points();
  auto vertex_offsets = tf::make_range(arrangement.vertex_offsets());
  auto triangle_tags = arrangement.triangle_tags();
  auto triangle_slots = arrangement.triangle_slots();
  auto pooled = arrangement.pooled_slots();
  auto corner = [&](const auto &v, Index own_tag) -> tf::point<Int, 3> {
    if (v.source == source::created)
      return created[std::size_t(v.id)];
    const auto tag =
        own_tag != Index(-1) ? own_tag : arrangement.tag_of_flat(v.id);
    return read_point(int(tag), v.id - vertex_offsets[std::size_t(tag)]);
  };

  tf::blocked_reduce(
      tf::enumerate(tris), bboxes, local_t{},
      [&](auto &&block, local_t &local) {
        local.clear();
        for (const auto &pair : block) {
          const auto &[e, tv] = pair;
          const Index c = triangle_labels[Index(e)];
          if (c == labels_t::none_label)
            continue;
          auto &box = local.touch(desc.bundle_of_component[c], init_bbox);
          const Index own_tag = pooled[triangle_slots[Index(e)]]
                                    ? Index(-1)
                                    : triangle_tags[Index(e)];
          for (std::size_t i = 0; i < 3; ++i)
            grow(box, corner(tv[i], own_tag));
        }
      },
      combine_bboxes);

  return bboxes;
}

} // namespace tf::csg::graph
