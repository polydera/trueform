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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./domain_inclusions.hpp"
#include "./sheet_sides.hpp"
#include "./triangle_component_labels.hpp"
#include "./winding_side.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief State every domain the volumes left unstated by the winding — or
///        by the sheet's own walls where every wall of it is the sheet's —
///        and publish the sheet columns of the inclusion table.
///
/// A wall's point is the centroid of one of its triangles, stated by its
/// corners, as the seeder states a bundle's.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename GetMeshPoint>
auto settle_sheet_sides(
    tf::csg::graph::domain_inclusions &inc,
    tf::csg::graph::sheet_sides<Index> &sides,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::offset_block_buffer<Index, Index> &domain_components,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form, const GetMeshPoint &get_mesh_point)
    -> void {
  using sides_t = tf::csg::graph::sheet_sides<Index>;
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;
  using source = tf::intersect::graph::vertex_source;
  using IntPt = tf::exact::pt3<Int>;
  const Index n_sheets = sides.n_sheets();
  const Index n_domains = desc.n_domains;
  if (n_sheets == Index(0) || n_domains == Index(0))
    return;

  tf::buffer<std::array<Index, 3>> asks; // (sheet, wall, domain)
  tf::generic_generate(
      tf::make_sequence_range(n_sheets * n_domains), asks,
      [&](Index pair, tf::buffer<std::array<Index, 3>> &buffer) {
        const Index k = pair / n_domains;
        const Index d = pair % n_domains;
        if (sides.state(k, d) != sides_t::unstated)
          return;
        for (const Index c : domain_components[std::size_t(d)])
          if (!sides.touches(desc, crossings, c, k)) {
            buffer.push_back({k, c, d});
            return;
          }
        sides.state(k, d) =
            sides.walled(desc, crossings, domain_components, k, d);
      });

  if (asks.size() > 0) {
    tbb::parallel_sort(asks.begin(), asks.end());
    const Index n_components = Index(desc.tag_of_component.size());
    tf::buffer<char> asked;
    asked.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_fill(asked, char(0));
    for (const auto &ask : asks)
      asked[ask[1]] = char(1);

    auto voffs = tf::make_range(arrangement.vertex_offsets());
    const auto &created = arrangement.created_points();
    auto point_of_vertex = [&](const auto &v, Index own_tag) -> IntPt {
      if (v.source == source::created)
        return created[std::size_t(v.id)];
      const auto tg =
          own_tag != Index(-1) ? own_tag : arrangement.tag_of_flat(v.id);
      return get_mesh_point(int(tg),
                            Index(v.id - voffs[std::size_t(tg)]));
    };

    // Every carrier of an asked wall proposes itself as (wall, id): an
    // uncut face by its slot in the tag-major face order, whole, so its
    // first fan triangle carries its interior; a piece of a cut face past
    // every face. The smallest id is the wall's point; a block walks its
    // carriers in order, so a run of one wall proposes its first alone.
    const Index n_tags = Index(sides.sheet_of_tag.size());
    tf::buffer<Index> face_offsets;
    face_offsets.push_back(Index(0));
    for (Index t = Index(0); t < n_tags; ++t)
      apply_to_form(t, [&](const auto &form) {
        face_offsets.push_back(face_offsets[std::size_t(t)] +
                               Index(form.faces().size()));
      });
    const Index n_faces = face_offsets[std::size_t(n_tags)];
    tf::buffer<std::array<Index, 2>> candidates;
    // an uncut face's wall is a surface component of its own form
    tf::buffer<char> asked_tag;
    asked_tag.allocate(static_cast<std::size_t>(n_tags));
    for (auto &a : asked_tag)
      a = char(0);
    for (const auto &ask : asks)
      if (desc.tag_of_component[ask[1]] >= Index(0))
        asked_tag[std::size_t(desc.tag_of_component[ask[1]])] = char(1);
    for (Index t = Index(0); t < n_tags; ++t) {
      if (!asked_tag[std::size_t(t)])
        continue;
      apply_to_form(t, [&](const auto &form) {
        auto polygon_labels = labels.polygon_labels(t);
        auto faces = form.faces();
        tf::generic_generate(
            tf::make_sequence_range(Index(faces.size())), candidates,
            [&](Index f, tf::buffer<std::array<Index, 2>> &buffer) {
              const Index c = polygon_labels[f];
              if (c == labels_t::none_label || !asked[c] ||
                  faces[f].size() < 3 ||
                  (buffer.size() && buffer[buffer.size() - 1][0] == c))
                return;
              buffer.push_back({c, face_offsets[std::size_t(t)] + f});
            },
            tf::checked);
      });
    }
    auto tris = arrangement.global().exposed_tris();
    auto tri_labels = labels.triangle_labels();
    tf::generic_generate(
        tf::make_sequence_range(Index(tris.size())), candidates,
        [&](Index e, tf::buffer<std::array<Index, 2>> &buffer) {
          const Index c = tri_labels[e];
          if (c != labels_t::none_label && asked[c] &&
              !(buffer.size() && buffer[buffer.size() - 1][0] == c))
            buffer.push_back({c, n_faces + e});
        },
        tf::checked);
    tbb::parallel_sort(candidates.begin(), candidates.end());
    candidates.erase_till_end(std::unique(
        candidates.begin(), candidates.end(),
        [](const auto &a, const auto &b) { return a[0] == b[0]; }));

    auto tri_tags = arrangement.triangle_tags();
    auto slots = arrangement.triangle_slots();
    auto pooled = arrangement.pooled_slots();
    tf::buffer<std::array<IntPt, 3>> point_of_wall;
    point_of_wall.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_for_each(
        tf::make_range(candidates),
        [&](const std::array<Index, 2> &candidate) {
          const Index c = candidate[0];
          const Index id = candidate[1];
          if (id >= n_faces) {
            const Index e = id - n_faces;
            const Index own = pooled[slots[e]] ? Index(-1) : tri_tags[e];
            point_of_wall[c] = {point_of_vertex(tris[e][0], own),
                                point_of_vertex(tris[e][1], own),
                                point_of_vertex(tris[e][2], own)};
          } else {
            const Index t =
                Index(std::upper_bound(face_offsets.begin(),
                                       face_offsets.end(), id) -
                      face_offsets.begin()) -
                Index(1);
            apply_to_form(t, [&](const auto &form) {
              auto face = form.faces()[id - face_offsets[std::size_t(t)]];
              point_of_wall[c] = {get_mesh_point(int(t), Index(face[0])),
                                  get_mesh_point(int(t), Index(face[1])),
                                  get_mesh_point(int(t), Index(face[2]))};
            });
          }
          asked[c] = char(2);
        },
        tf::checked);

    // one winding pass per sheet over the distinct walls asked of it
    tf::buffer<std::array<IntPt, 3>> queries;
    tf::buffer<Index> query_of_ask;
    query_of_ask.allocate(asks.size());
    for (std::size_t lo = 0; lo < asks.size();) {
      const Index k = asks[lo][0];
      std::size_t hi = lo;
      queries.clear();
      for (; hi < asks.size() && asks[hi][0] == k; ++hi) {
        const Index c = asks[hi][1];
        if (asked[c] != char(2)) {
          query_of_ask[hi] = Index(-1);
          continue;
        }
        if (queries.size() == 0 || asks[hi][1] != asks[hi - 1][1])
          queries.push_back(point_of_wall[c]);
        query_of_ask[hi] = Index(queries.size() - 1);
      }
      const Index t = sides.tags[k];
      apply_to_form(t, [&](const auto &form) {
        auto bits = tf::csg::graph::winding_side<Int>(form, t, get_mesh_point,
                                                      queries);
        for (std::size_t j = lo; j < hi; ++j) {
          const Index q = query_of_ask[j];
          const Index d = asks[j][2];
          sides.state(k, d) =
              q < Index(0) ? sides.walled(desc, crossings, domain_components,
                                          k, d)
              : bits[std::size_t(q)] ? sides_t::behind
                                     : sides_t::front;
        }
      });
      lo = hi;
    }
  }

  tf::parallel_for_each(
      tf::make_sequence_range(n_domains),
      [&](Index d) {
        for (Index k = Index(0); k < n_sheets; ++k) {
          if (sides.state(k, d) == sides_t::behind)
            inc.set(std::size_t(d), std::size_t(sides.tags[k]));
          else
            inc.clear(std::size_t(d), std::size_t(sides.tags[k]));
        }
      },
      tf::checked);
}

} // namespace tf::csg::graph
