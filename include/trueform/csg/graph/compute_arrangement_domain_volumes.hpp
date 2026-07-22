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
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../cut/arrangements/component_labels.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include "../../cut/region_triangulator.hpp"
#include "../../exact/determinant.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Per-domain signed volumes of the implicit arrangement.
///
/// Sums fan-triangulated Gauss `det(p0, p_i, p_{i+1})` (T2 width) per
/// component, then redistributes with the
/// @ref tf::topology::domains::compute_domain_volumes sign convention:
/// `domain_of_side[2c + 0]` gets `-C`, `domain_of_side[2c + 1]` gets
/// `+C`. Both surface sources are walked: uncut faces per form via
/// `ag.polygon_labels(t)`, cut loops via `ag.triangle_labels()`. Dead /
/// dropped components (`none_label`) are skipped.
template <typename Forms, typename Index, typename Int, typename GetPoint>
auto compute_arrangement_domain_volumes(
    const Forms &tagged_forms, const tf::cut::component_labels<Index> &ag,
    const tf::cut::region_triangulator<Index, Int> &rt,
    const tf::cut::arrangement_descriptor<Index> &desc,
    const GetPoint &get_point)
    -> tf::buffer<typename tf::exact::meta<Int>::T2> {
  using AccumT = typename tf::exact::meta<Int>::T2;
  using ag_t = tf::cut::component_labels<Index>;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  const Index n_components = ag.n_components();
  const Index n_domains = desc.n_domains;
  const Index n_tags = ag.n_tags();

  tf::buffer<AccumT> volumes;
  volumes.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_fill(volumes, AccumT(0));

  if (n_components == 0 || n_domains == 0)
    return volumes;

  tf::buffer<AccumT> contributions;
  contributions.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_fill(contributions, AccumT(0));

  auto fresh_prototype = [&] {
    tf::buffer<AccumT> p;
    p.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_fill(p, AccumT(0));
    return p;
  };

  auto combine = [](const tf::buffer<AccumT> &local, tf::buffer<AccumT> &out) {
    for (decltype(out.size()) i = 0; i < out.size(); ++i)
      out[i] += local[i];
  };

  // Uncut faces, per form.
  for (Index t = Index(0); t < n_tags; ++t) {
    auto labels = ag.polygon_labels(t);
    auto faces = tagged_forms[t].faces();
    if (faces.size() == 0)
      continue;
    auto prototype = fresh_prototype();
    tf::blocked_reduce(
        tf::enumerate(faces), contributions, prototype,
        [&](auto &&block, tf::buffer<AccumT> &local) {
          for (const auto &pair : block) {
            const auto &[f, face] = pair;
            const Index c = labels[Index(f)];
            if (c == ag_t::none_label)
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
            local[c] += sum;
          }
        },
        combine);
  }

  // Cut loops.
  if (rt.loops().size() > 0) {
    auto loops = rt.loops();
    auto descs = rt.descriptors();
    auto loop_labels = ag.triangle_labels();
    auto prototype = fresh_prototype();
    tf::blocked_reduce(
        tf::enumerate(loops), contributions, prototype,
        [&](auto &&block, tf::buffer<AccumT> &local) {
          for (const auto &pair : block) {
            const auto &[l, loop] = pair;
            const Index c = loop_labels[Index(l)];
            if (c == ag_t::none_label)
              continue;
            const auto n_lv = loop.size();
            if (n_lv < 3)
              continue;
            const Index tag = descs[Index(l)].tag;
            if (tag == Index(-1))
              continue;
            auto p0 = get_point(loop[0], tag);
            AccumT sum(0);
            for (std::size_t i = 1; i + 1 < n_lv; ++i) {
              auto p1 = get_point(loop[i], tag);
              auto p2 = get_point(loop[i + 1], tag);
              sum += tf::exact::determinant_value(p0, p1, p2);
            }
            local[c] += sum;
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

} // namespace tf::csg::graph
