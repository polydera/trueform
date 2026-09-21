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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./triangle_component_labels.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per-component crossing layers: the forms a wall carries, each
///        with the signed count it adds when the wall is crossed
///        `side 0 -> side 1`.
///
/// A coincident stack states the wall of several forms at once. The
/// arrangement keeps one survivor triangle and records every duplicate as
/// a coplanar triple whose `opposing` flag states the duplicate's winding
/// against the survivor's, so a wall's layers are the survivor's own tag
/// at `+1` plus one entry per duplicate, `-1` where it opposes. Equal tags
/// collapse into one entry carrying their sum: an opposing coincident pair
/// carries zero and an aligned one two.
///
/// Tessellation and refinement multiply triangles, not layers, so the
/// stack is read at ONE representative triangle per component. Almost no
/// component carries a stack at all, so only the ones that do are stored —
/// `components` ascending, binary-searched by @ref row_of — and every
/// other component's crossing is its own tag at unit, which the reader
/// already holds.
///
/// A representative stands for its component only if every live carrier of
/// that component states the same stack. That is verified with complete
/// coverage — every survivor group against its component's, and every
/// stackless live carrier of a stacked component — and a component failing
/// it is REPORTED, never refused: `n_defects` counts them,
/// `defect_component` names the first, and classification continues on the
/// representatives.
template <typename Index> struct component_crossings {
  /// The components carrying a stack, ascending.
  tf::buffer<Index> components;
  /// Per entry of `components`: its `(tag, signed count)` layers,
  /// ascending by tag, one entry per distinct tag.
  tf::offset_block_buffer<Index, std::array<Index, 2>> layers;
  Index n_defects = 0;
  Index defect_component = Index(-1);

  /// @brief The row of component `c`, `-1` when it carries no stack.
  auto row_of(Index c) const -> Index {
    const auto first = components.begin();
    const auto last = components.end();
    const auto it = std::lower_bound(first, last, c);
    return it != last && *it == c ? Index(it - first) : Index(-1);
  }
};

/// @ingroup csg_graph_internals
/// @brief Read every stacked component's layers off a representative and
///        verify the representative stands for the whole component.
template <typename Index, typename Arrangement>
auto make_component_crossings(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels)
    -> component_crossings<Index> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;

  component_crossings<Index> out;
  const Index n_components = labels.n_components();
  auto triples = arrangement.coplanar_triples();
  if (n_components == Index(0) || triples.size() == 0)
    return out;

  auto tri_labels = labels.triangle_labels();
  auto tri_tags = arrangement.triangle_tags();

  // The stacked components take their rows in ascending order, so the
  // product is the sorted table `row_of` binary-searches.
  tf::buffer<Index> row_of_component;
  row_of_component.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_fill(row_of_component, Index(-1));
  const Index pending = Index(-2);
  for (const auto &triple : triples) {
    const Index c = tri_labels[triple.survivor];
    if (c != labels_t::none_label)
      row_of_component[c] = pending;
  }
  for (Index c = Index(0); c < n_components; ++c)
    if (row_of_component[c] == pending) {
      row_of_component[c] = Index(out.components.size());
      out.components.push_back(c);
    }
  if (out.components.size() == 0)
    return out;

  // The triples arrive sorted by (survivor, dead), so a survivor's whole
  // stack is one contiguous run.
  auto for_each_stack = [&](auto &&body) {
    const std::size_t n = triples.size();
    for (std::size_t lo = 0; lo < n;) {
      std::size_t hi = lo;
      while (hi < n && triples[hi].survivor == triples[lo].survivor)
        ++hi;
      body(triples[lo].survivor, lo, hi);
      lo = hi;
    }
  };

  tf::buffer<std::array<Index, 2>> rep_range; // [lo, hi) into the triples
  rep_range.allocate(out.components.size());
  tf::parallel_fill(rep_range, std::array<Index, 2>{Index(-1), Index(-1)});
  for_each_stack([&](Index survivor, std::size_t lo, std::size_t hi) {
    const Index c = tri_labels[survivor];
    if (c == labels_t::none_label)
      return;
    auto &rep = rep_range[static_cast<std::size_t>(row_of_component[c])];
    if (rep[0] == Index(-1))
      rep = {Index(lo), Index(hi)};
  });

  tf::buffer<std::array<Index, 2>> stack;
  auto normalize = [&](Index survivor, std::size_t lo, std::size_t hi) {
    stack.clear();
    stack.push_back({tri_tags[survivor], Index(0)});
    for (std::size_t k = lo; k < hi; ++k)
      stack.push_back({tri_tags[triples[k].dead], Index(triples[k].opposing)});
    std::sort(stack.begin(), stack.end());
  };

  tf::offset_block_buffer<Index, std::array<Index, 2>> rep_layers;
  tf::buffer<std::array<Index, 2>> block;
  for (std::size_t r = 0; r < out.components.size(); ++r) {
    const auto rep = rep_range[r];
    normalize(triples[static_cast<std::size_t>(rep[0])].survivor,
              static_cast<std::size_t>(rep[0]),
              static_cast<std::size_t>(rep[1]));
    rep_layers.push_back(tf::make_range(stack));
    block.clear();
    for (const auto &layer : stack) {
      const Index delta = layer[1] ? Index(-1) : Index(1);
      if (block.size() > 0 && block[block.size() - 1][0] == layer[0])
        block[block.size() - 1][1] += delta;
      else
        block.push_back({layer[0], delta});
    }
    out.layers.push_back(tf::make_range(block));
  }

  tf::buffer<Index> defects;
  for_each_stack([&](Index survivor, std::size_t lo, std::size_t hi) {
    const Index c = tri_labels[survivor];
    if (c == labels_t::none_label)
      return;
    const Index r = row_of_component[c];
    if (rep_range[static_cast<std::size_t>(r)][0] == Index(lo))
      return;
    normalize(survivor, lo, hi);
    const auto rep = rep_layers[static_cast<std::size_t>(r)];
    if (stack.size() != rep.size() ||
        !std::equal(stack.begin(), stack.end(), rep.begin()))
      defects.push_back(c);
  });

  // Coverage: a live carrier of a stacked component that states no stack
  // of its own would hide behind the representative. The cut tier answers
  // by the triples' own sorted survivors; an uncut face has no triple at
  // all, so reaching one is already the answer.
  auto carries_stack = [&](Index e) -> bool {
    const auto first = triples.begin();
    const auto last = triples.end();
    const auto it = std::lower_bound(
        first, last, e,
        [](const auto &triple, Index id) { return triple.survivor < id; });
    return it != last && it->survivor == e;
  };
  tf::generic_generate(tf::make_sequence_range(Index(tri_labels.size())),
                       defects, [&](Index e, tf::buffer<Index> &found) {
                         const Index c = tri_labels[e];
                         if (c == labels_t::none_label ||
                             row_of_component[c] == Index(-1))
                           return;
                         if (!carries_stack(e))
                           found.push_back(c);
                       });
  for (Index t = Index(0); t < arrangement.n_tags(); ++t) {
    auto polygon_labels = labels.polygon_labels(t);
    tf::generic_generate(tf::make_sequence_range(Index(polygon_labels.size())),
                         defects, [&](Index f, tf::buffer<Index> &found) {
                           const Index c = polygon_labels[f];
                           if (c != labels_t::none_label &&
                               row_of_component[c] != Index(-1))
                             found.push_back(c);
                         });
  }

  std::sort(defects.begin(), defects.end());
  defects.erase_till_end(std::unique(defects.begin(), defects.end()));
  out.n_defects = Index(defects.size());
  if (defects.size() > 0)
    out.defect_component = defects[0];
  return out;
}

} // namespace tf::csg::graph
