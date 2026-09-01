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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/sparse_block_accumulator.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../exact/determinant.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./arrangement_descriptor.hpp"
#include "./triangle_component_labels.hpp"

#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per-domain signed volumes of the implicit arrangement.
///
/// Sums the Gauss determinant `det(p0, p_i, p_{i+1})` (T2 width) over
/// each uncut face's fan and each cut triangle, per component, then
/// redistributes with the
/// @ref tf::topology::domains::compute_domain_volumes sign convention:
/// `domain_of_side[2c + 0]` gets `-C`, `domain_of_side[2c + 1]` gets
/// `+C`. Both surface sources are walked: uncut faces per form via
/// `labels.polygon_labels(t)`, the cut-triangle stream via
/// `labels.triangle_labels()`. Dead / dropped components
/// (`none_label`) are skipped, as are triangles whose form tag is
/// `-1`.
///
/// The two carriers live in different vertex-id spaces, so each has its
/// own point authority: `get_point` resolves a form-local original
/// vertex of `tag`, `get_corner_point` resolves a stream corner.
///
/// @tparam Int    The lattice integer the points are expressed in.
/// @param labels  The label tier: `n_components()`, `polygon_labels(t)`,
///                `triangle_labels()`, `none_label`.
/// @param n_tags  The number of forms in the arrangement.
/// @param triangles The cut-triangle stream, one triangle per element.
/// @param tags    Per-stream-triangle form tags.
template <typename Int, typename Index, typename ApplyToForm, typename Labels,
          typename Triangles, typename Tags, typename GetPoint,
          typename GetCornerPoint>
auto compute_arrangement_domain_volumes(
    const ApplyToForm &apply_to_form, const Labels &labels,
    const tf::csg::graph::arrangement_descriptor<Index> &desc, Index n_tags,
    const Triangles &triangles, const Tags &tags, const GetPoint &get_point,
    const GetCornerPoint &get_corner_point)
    -> tf::buffer<typename tf::exact::meta<Int>::T2> {
  using AccumT = typename tf::exact::meta<Int>::T2;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  const Index n_components = labels.n_components();
  const Index n_domains = desc.n_domains;

  tf::buffer<AccumT> volumes;
  volumes.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_fill(volumes, AccumT(0));

  if (n_components == 0 || n_domains == 0)
    return volumes;

  tf::buffer<AccumT> contributions;
  contributions.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_fill(contributions, AccumT(0));

  // THE COMPONENT AXIS IS GLOBAL AND A BLOCK TOUCHES A RUN OF IT, so the
  // block carries only the components it names. Sums are exact integers, so
  // the order they are added in cannot change a volume.
  using local_t = tf::sparse_block_accumulator<Index, AccumT>;
  auto combine = [](const local_t &local, tf::buffer<AccumT> &out) {
    for (const auto &entry : local.entries)
      out[entry.key] += entry.value;
  };

  // Uncut faces, per form.
  for (Index t = Index(0); t < n_tags; ++t)
    apply_to_form(t, [&](const auto &tagged_form) {
      auto face_labels = labels.polygon_labels(t);
      auto faces = tagged_form.faces();
      if (faces.size() == 0)
        return;
      tf::blocked_reduce(
          tf::enumerate(faces), contributions, local_t{},
          [&](auto &&block, local_t &local) {
            local.clear();
            for (const auto &pair : block) {
              const auto &[f, face] = pair;
              const Index c = face_labels[Index(f)];
              if (c == Labels::none_label)
                continue;
              const auto n_fv = face.size();
              if (n_fv < 3)
                continue;
              auto p0 = get_point(
                  vertex_t{source::original, Index(face[0]),
                           {short(0), tf::topo_type::vertex}},
                  t);
              AccumT sum(0);
              for (std::size_t i = 1; i + 1 < n_fv; ++i) {
                auto p1 = get_point(
                    vertex_t{source::original, Index(face[i]),
                             {short(i), tf::topo_type::vertex}},
                    t);
                auto p2 = get_point(
                    vertex_t{source::original, Index(face[i + 1]),
                             {short(i + 1), tf::topo_type::vertex}},
                    t);
                sum += tf::exact::determinant_value(p0, p1, p2);
              }
              local.touch(c, AccumT(0)) += sum;
            }
          },
          combine);
    });

  // Cut triangles.
  if (triangles.size() > 0) {
    auto triangle_labels = labels.triangle_labels();
    tf::blocked_reduce(
        tf::enumerate(triangles), contributions, local_t{},
        [&](auto &&block, local_t &local) {
          local.clear();
          for (const auto &pair : block) {
            const auto &[l, tri] = pair;
            const Index c = triangle_labels[Index(l)];
            if (c == Labels::none_label)
              continue;
            const Index tag = tags[Index(l)];
            if (tag == Index(-1))
              continue;
            auto p0 = get_corner_point(tri[0], tag);
            auto p1 = get_corner_point(tri[1], tag);
            auto p2 = get_corner_point(tri[2], tag);
            local.touch(c, AccumT(0)) +=
                tf::exact::determinant_value(p0, p1, p2);
          }
        },
        combine);
  }

  // Redistribute per-component into per-domain (sequential — parallel
  // scatter would need atomics on the shared domain slots).
  auto domain_of_side = desc.domain_of_side;
  for (Index c = Index(0); c < n_components; ++c) {
    const auto &C = contributions[c];
    volumes[domain_of_side[2 * c + 0]] -= C;
    volumes[domain_of_side[2 * c + 1]] += C;
  }

  return volumes;
}

/// @ingroup csg_graph_internals
/// @brief Triangle-grain volumes: the two carriers are disjoint —
///        uncut faces walk `polygon_labels` in each form's local vertex
///        id space, the exposed stream (cut faces only) walks
///        `triangle_labels` with corners in the created / flat-original
///        space.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename GetMeshPoint>
auto compute_arrangement_domain_volumes(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const ApplyToForm &apply_to_form, const GetMeshPoint &get_mesh_point)
    -> tf::buffer<typename tf::exact::meta<Int>::T2> {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  const auto &created = arrangement.created_points();
  auto vertex_offsets = tf::make_range(arrangement.vertex_offsets());

  return compute_arrangement_domain_volumes<Int>(
      apply_to_form, labels, desc, arrangement.n_tags(), arrangement.global().exposed_tris(),
      arrangement.triangle_tags(),
      [&get_mesh_point](const vertex_t &v, Index tag) -> tf::point<Int, 3> {
        return get_mesh_point(int(tag), v.id);
      },
      [&created, &arrangement, vertex_offsets,
       &get_mesh_point](const vertex_t &v, Index) -> tf::point<Int, 3> {
        if (v.source == source::created)
          return created[std::size_t(v.id)];
        const auto tag = arrangement.tag_of_flat(v.id);
        return get_mesh_point(int(tag),
                              Index(v.id - vertex_offsets[std::size_t(tag)]));
      });
}

} // namespace tf::csg::graph
