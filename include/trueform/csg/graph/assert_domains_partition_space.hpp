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
#include "../../core/algorithm/sparse_block_accumulator.hpp"
#include "../../core/buffer.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/point.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./triangle_component_labels.hpp"
#include "./union_find.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// The walls partition space: every region they bound is closed, its
/// boundary's area vector zero. That is the volume law stated exactly — a
/// region's volume does not depend on the origin it is summed about exactly
/// when its boundary closes. The signed volumes of all domains sum to zero by
/// construction, each wall adding to one side what it takes from the other,
/// so that sum states nothing. A sheet bounds nothing, and neither does a
/// wall whose every stack layer nets to zero — one form's surface folded
/// onto itself, its survivor standing for both faces of a region of no
/// thickness — so the domains either parts are read as one region. A failed
/// plane states its holes and a lone page is never walked, so either disarms
/// the check.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename ReadPoint>
auto assert_domains_partition_space(
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::buffer<char> &is_sheet, const ApplyToForm &apply_to_form,
    const ReadPoint &read_point) -> void {
  if (arrangement.failed().size() != 0 || desc.n_invalid_fans != Index(0))
    return;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using area_t = std::array<T2, 3>;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;
  const Index n_components = labels.n_components();
  const auto n_domains = std::size_t(desc.n_domains);
  if (n_components == Index(0) || n_domains == 0)
    return;

  const auto add_edge = [](area_t &area, const tf::point<Int, 3> &a,
                           const tf::point<Int, 3> &b) {
    area[0] += T2(T1(a[1]) * b[2]) - T1(a[2]) * b[1];
    area[1] += T2(T1(a[2]) * b[0]) - T1(a[0]) * b[2];
    area[2] += T2(T1(a[0]) * b[1]) - T1(a[1]) * b[0];
  };
  tf::buffer<area_t> areas;
  areas.allocate(std::size_t(n_components));
  for (auto &area : areas)
    area = area_t{T2(0), T2(0), T2(0)};
  using local_t = tf::sparse_block_accumulator<Index, area_t>;
  const auto combine = [](const local_t &local, tf::buffer<area_t> &out) {
    for (const auto &entry : local.entries)
      for (int k = 0; k < 3; ++k)
        out[entry.key][k] += entry.value[k];
  };
  const area_t zero{T2(0), T2(0), T2(0)};

  for (Index t = Index(0); t < arrangement.n_tags(); ++t)
    apply_to_form(t, [&](const auto &tagged_form) {
      const auto face_labels = labels.polygon_labels(t);
      const auto faces = tagged_form.faces();
      if (faces.size() == 0)
        return;
      tf::blocked_reduce(
          tf::enumerate(faces), areas, local_t{},
          [&](auto &&block, local_t &local) {
            local.clear();
            for (const auto &pair : block) {
              const auto &[f, face] = pair;
              const Index c = face_labels[Index(f)];
              if (c == labels.none_label)
                continue;
              auto &area = local.touch(c, zero);
              const auto n = std::size_t(face.size());
              for (std::size_t k = 0; k < n; ++k)
                add_edge(area, read_point(int(t), Index(face[k])),
                         read_point(int(t), Index(face[(k + 1) % n])));
            }
          },
          combine);
    });

  const auto &created = arrangement.created_points();
  const auto &vertex_offsets = arrangement.vertex_offsets();
  const auto corner = [&](const vertex_t &v) -> tf::point<Int, 3> {
    if (v.source == source::created)
      return created[std::size_t(v.id)];
    const auto tag = arrangement.tag_of_flat(v.id);
    return read_point(int(tag),
                      Index(v.id - vertex_offsets[std::size_t(tag)]));
  };
  const auto triangles = arrangement.global().exposed_tris();
  const auto tags = arrangement.triangle_tags();
  const auto triangle_labels = labels.triangle_labels();
  if (triangles.size() > 0)
    tf::blocked_reduce(
        tf::enumerate(triangles), areas, local_t{},
        [&](auto &&block, local_t &local) {
          local.clear();
          for (const auto &pair : block) {
            const auto &[e, tri] = pair;
            const Index c = triangle_labels[Index(e)];
            if (c == labels.none_label || tags[Index(e)] == Index(-1))
              continue;
            auto &area = local.touch(c, zero);
            for (int k = 0; k < 3; ++k)
              add_edge(area, corner(tri[k]), corner(tri[(k + 1) % 3]));
          }
        },
        combine);

  tf::csg::graph::union_find<Index> regions;
  regions.reset(n_domains);
  for (Index c = Index(0); c < n_components; ++c) {
    const Index t = desc.tag_of_component[c];
    if (t >= Index(0) && std::size_t(t) < is_sheet.size() && is_sheet[t])
      regions.unite(desc.domain_of_side[2 * c], desc.domain_of_side[2 * c + 1]);
  }
  for (std::size_t row = 0; row < crossings.components.size(); ++row) {
    bool net_zero = true;
    for (const auto &layer : crossings.layers[row])
      net_zero = net_zero && layer[1] == Index(0);
    const Index c = crossings.components[row];
    if (net_zero)
      regions.unite(desc.domain_of_side[2 * c], desc.domain_of_side[2 * c + 1]);
  }
  tf::buffer<area_t> boundary;
  boundary.allocate(n_domains);
  for (auto &area : boundary)
    area = zero;
  for (Index c = Index(0); c < n_components; ++c) {
    auto &outside = boundary[regions.find(desc.domain_of_side[2 * c])];
    auto &inside = boundary[regions.find(desc.domain_of_side[2 * c + 1])];
    for (int k = 0; k < 3; ++k) {
      outside[k] -= areas[c][k];
      inside[k] += areas[c][k];
    }
  }
  bool closed = true;
  for (const auto &area : boundary)
    closed = closed && area[0] == T2(0) && area[1] == T2(0) && area[2] == T2(0);
  tf::core::verify(closed, "a region the walls bound is not closed");
}

} // namespace tf::csg::graph
