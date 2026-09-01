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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./triangle_component_labels.hpp"
#include "./arrangement_descriptor.hpp"
#include "./domain_inclusions.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Multi-source XOR-BFS that propagates inclusion bits across
///        the implicit domain graph from pre-seeded domains.
///
/// At each step, every domain `d_other` reached from a visited domain
/// `d` via component `c` gets
/// `inclusion[d_other] = inclusion[d] XOR B(c)`, where `B(c)` is the
/// OR'd tag mask for `c` (its own form bit plus any coplanar pack
/// member's tag). The seed bits are kept as-is — see
/// @ref tf::csg::graph::seed_inclusion_bits for where they come from.
///
/// @param inc          Per-domain inclusion bitvectors; pre-seeded at
///                     every domain id in `seeds`.
/// @param desc         Arrangement descriptor (`domain_of_side`,
///                     `n_domains`, `tag_of_component`).
/// @param n_components The component count `labels` maps into.
/// @param labels       Per-carrier component labels; dead carriers hold
///                     `component_labels::none_label`.
/// @param tags         Per-carrier form tags.
/// @param coplanar     Coplanar pack records read as
///                     `{survivor, dead, reversed}`.
/// @param seeds        Pre-seeded domain ids — each becomes a BFS root.
template <typename Index, typename Labels, typename Tags, typename Coplanar>
auto propagate_inclusion_bits(
    tf::csg::graph::domain_inclusions &inc,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    Index n_components, const Labels &labels, const Tags &tags,
    const Coplanar &coplanar, const tf::buffer<Index> &seeds) -> void {
  const Index n_domains = desc.n_domains;
  const std::size_t words_per_domain = inc.words_per_domain;

  if (n_components == 0 || n_domains == 0 || seeds.size() == 0)
    return;

  using labels_t = tf::csg::graph::triangle_component_labels<Index>;
  auto domain_of_side_buf = desc.domain_of_side;

  // B(c) bitmask: each component's own tag, plus the tags of every
  // coplanar duplicate folded into it.
  tf::buffer<std::uint32_t> B;
  B.allocate(static_cast<std::size_t>(n_components) * words_per_domain);
  tf::parallel_fill(B, std::uint32_t(0));

  auto carry = [&](Index c, Index t) {
    B[static_cast<std::size_t>(c) * words_per_domain +
      static_cast<std::size_t>(t) / 32u] |=
        std::uint32_t(1) << (static_cast<unsigned>(t) % 32u);
  };

  auto tag_of_component = desc.tag_of_component;
  tf::parallel_for_each(
      tf::make_sequence_range(n_components), [&](Index c) {
        const Index t = tag_of_component[c];
        if (t == Index(-1))
          return;
        carry(c, t);
      });

  for (const auto &record : coplanar) {
    const Index survivor = record[0];
    const Index dead = record[1];
    const Index c = labels[survivor];
    if (c == labels_t::none_label)
      continue;
    carry(c, tags[dead]);
  }

  // Per-domain → incident components via counting sort.
  tf::buffer<Index> in_offsets;
  in_offsets.allocate(static_cast<std::size_t>(n_domains + 1));
  tf::parallel_fill(in_offsets, Index(0));

  for (Index c = 0; c < n_components; ++c) {
    in_offsets[domain_of_side_buf[2 * c + 0] + 1]++;
    in_offsets[domain_of_side_buf[2 * c + 1] + 1]++;
  }
  for (Index i = 0; i < n_domains; ++i)
    in_offsets[i + 1] += in_offsets[i];

  tf::buffer<Index> in_data;
  in_data.allocate(static_cast<std::size_t>(in_offsets[n_domains]));
  {
    tf::buffer<Index> cursor;
    cursor.allocate(static_cast<std::size_t>(n_domains));
    tf::parallel_for_each(
        tf::enumerate(cursor),
        [&](auto pair) {
          auto &&[i, c] = pair;
          c = in_offsets[i];
        },
        tf::checked);
    for (Index c = 0; c < n_components; ++c) {
      const Index d0 = domain_of_side_buf[2 * c + 0];
      const Index d1 = domain_of_side_buf[2 * c + 1];
      in_data[cursor[d0]++] = c;
      in_data[cursor[d1]++] = c;
    }
  }

  // Multi-source BFS-XOR from `seeds`.
  tf::buffer<char> visited;
  visited.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_fill(visited, char(0));

  tf::buffer<Index> q;
  std::size_t head = 0;
  for (auto s : seeds) {
    if (s < Index(0) || s >= n_domains)
      continue;
    if (visited[s])
      continue;
    visited[s] = char(1);
    q.push_back(s);
  }

  while (head < q.size()) {
    const Index d = q[head++];
    const Index lo = in_offsets[d];
    const Index hi = in_offsets[d + 1];
    for (Index k = lo; k < hi; ++k) {
      const Index c = in_data[k];
      const Index d0 = domain_of_side_buf[2 * c + 0];
      const Index d1 = domain_of_side_buf[2 * c + 1];
      const Index d_other = (d == d0) ? d1 : d0;
      if (visited[d_other])
        continue;
      for (std::size_t w = 0; w < words_per_domain; ++w)
        inc.bits[static_cast<std::size_t>(d_other) * words_per_domain + w] =
            inc.bits[static_cast<std::size_t>(d) * words_per_domain + w] ^
            B[static_cast<std::size_t>(c) * words_per_domain + w];
      visited[d_other] = char(1);
      q.push_back(d_other);
    }
  }
}

/// @ingroup csg_graph_internals
/// @brief Triangle-grain propagation: carriers are the arrangement's
///        exposed triangles, labelled by
///        @ref tf::csg::graph::triangle_component_labels.
template <typename Index, typename Arrangement>
auto propagate_inclusion_bits(
    tf::csg::graph::domain_inclusions &inc,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const tf::buffer<Index> &seeds) -> void {
  using triple_t = typename Arrangement::coplanar_triple;
  propagate_inclusion_bits(
      inc, desc, labels.n_components(), labels.triangle_labels(),
      arrangement.triangle_tags(),
      tf::make_mapped_range(arrangement.coplanar_triples(),
                            [](const triple_t &t) {
                              return std::array<Index, 3>{t.survivor, t.dead,
                                                          Index(t.opposing)};
                            }),
      seeds);
}

} // namespace tf::csg::graph
