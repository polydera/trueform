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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/set_component_labels.hpp"
#include "./partition_labels.hpp"
#include "tbb/parallel_sort.h"

namespace tf::cut {

/// Build set_component_labels for one mesh.
///
/// Merges cut face components connected through same-tag manifold edges,
/// and detects open/closed sets. Used for classifying missing components
/// (no intersection edges → point-in-mesh fallback).
template <typename LabelType, typename Index, typename Policy,
          typename Descriptors, typename Connectivity, typename AllDescs>
auto make_set_component_labels(
    const tf::polygons<Policy> &polygons, const Descriptors &descriptors,
    const Connectivity &conn, const AllDescs &all_descs,
    const tf::cut::partition_labels<LabelType> &pal, Index tag_offset) {
  tf::buffer<std::array<LabelType, 2>> ids;
  tf::buffer<LabelType> labels;
  labels.allocate(pal.polygon_labels.size());
  tf::generic_generate(
      tf::zip(descriptors, pal.cut_labels, conn), ids,
      tf::hash_set<std::array<LabelType, 2>, tf::array_hash<LabelType, 2>>{},
      [&](const auto &tup, auto &buffer, auto &set) {
        auto &&[d, label, edge_conn] = tup;
        labels[d.object] = label;
        for (auto nexts : edge_conn) {
          if (std::count_if(nexts.begin(), nexts.end(),
                            [&d = d, &all_descs](auto x) {
                              return all_descs[x].tag == d.tag;
                            }) != 1)
            continue;
          for (auto next : nexts) {
            if (d.tag != all_descs[next].tag)
              continue;
            auto next_label = pal.cut_labels[next - tag_offset];
            if (next_label == -1)
              continue;
            std::array<LabelType, 2> pair{next_label, label};
            if (pair[0] == pair[1])
              continue;
            if (pair[0] > pair[1])
              std::swap(pair[0], pair[1]);
            if (set.insert(pair).second)
              buffer.push_back(pair);
          }
        }
      });
  tbb::parallel_sort(ids.begin(), ids.end());
  ids.erase_till_end(std::unique(ids.begin(), ids.end()));

  tf::buffer<LabelType> map;
  map.allocate(pal.n_components);
  LabelType n_labels = tf::make_dense_equivalence_class_map(ids, map);
  tf::buffer<tf::set_type> sets;
  sets.allocate(n_labels);
  tf::parallel_fill(sets, tf::set_type::closed);
  tf::parallel_for_each(
      tf::zip(polygons.manifold_edge_link(), labels, pal.polygon_labels),
      [&](auto pair) {
        auto &[mel, label, old_label] = pair;
        if (old_label != -1)
          label = map[old_label];
        else
          label = map[label];
        for (auto h : mel)
          if (h.is_boundary())
            sets[label] = tf::set_type::open;
      },
      tf::checked);
  return tf::set_component_labels<LabelType>{{std::move(labels), n_labels},
                                             std::move(sets)};
}

} // namespace tf::cut
