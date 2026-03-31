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
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/label_connected_components.hpp"
#include "../../topology/make_applier.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../../topology/topo_type.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"
#include "./partition_labels.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

namespace tf::cut {

/// Label connected components of uncut surface faces.
/// Faces that appear in descriptors are masked out; remaining faces
/// are flood-filled through manifold_edge_link.
template <typename Index, typename LabelType, typename Policy,
          typename Descriptors>
auto make_surface_component_labels2(const tf::polygons<Policy> &polygons,
                                   const Descriptors &descriptors,
                                   Index expected_components = 2) {
  static_assert(tf::has_manifold_edge_link_policy<Policy>,
                "Use polygons | tf::tag(manifold_edge_link)");
  tf::buffer<char> mask;
  mask.allocate(polygons.faces().size());
  tf::connected_component_labels<LabelType> cl;
  cl.labels.allocate(mask.size());
  tf::parallel_fill(mask, true);
  tf::parallel_for_each(descriptors, [&](const auto &desc) {
    mask[desc.object] = false;
    cl.labels[desc.object] = -1;
  });
  cl.n_components = tf::label_connected_components_masked(
      cl.labels, mask, tf::make_applier(polygons.manifold_edge_link()),
      expected_components);
  return cl;
}

/// Label connected components of cut face loops.
/// Walks through cut_graph connectivity, not crossing intersection edges.
/// Takes pre-sliced loops and connectivity for a single tag.
template <typename LabelType, typename Index, typename Loops,
          typename Connectivity, typename Descs>
auto make_cut_component_labels(const Loops &loops, const Connectivity &conn,
                               const tf::cut_graph<Index> &cg,
                               const Descs &descs) {
  tf::connected_component_labels<LabelType> cl;
  cl.labels.allocate(loops.size());
  cl.n_components = tf::label_connected_components<Index>(
      cl.labels, [&](Index id, const auto &f) {
        auto loop = loops[id];
        auto edge_conn = conn[id];
        auto my_tag = descs[id].tag;
        auto size = loop.size();
        auto prev = size - 1;
        for (decltype(size) i = 0; i < size; prev = i++) {
          if (cg.is_intersection_edge(loop[prev], loop[i]))
            continue;
          auto same_tag_count = 0;
          Index same_tag_neighbor = -1;
          for (auto nid : edge_conn[prev]) {
            if (descs[nid].tag == my_tag) {
              same_tag_count++;
              same_tag_neighbor = nid;
            }
          }
          if (same_tag_count == 1)
            f(same_tag_neighbor);
        }
      });
  return cl;
}

/// Label cut face components, then partition labels by tag.
/// Returns one connected_component_labels per tag.
template <typename LabelType, typename Index, typename Int>
auto make_cut_component_labels(const tf::face_cuts<Index, Int> &fc,
                               const tf::cut_graph<Index> &cg) {
  auto loops = fc.loops();
  auto conn = cg.connectivity_per_face_edge(fc);
  auto descs = fc.descriptors();
  auto cl = make_cut_component_labels<LabelType, Index>(loops, conn, cg, descs);

  auto per_tag = tf::make_offset_block_range(fc.tag_offsets(), cl.labels);

  tf::buffer<LabelType> map;
  map.allocate(cl.n_components);
  tf::parallel_fill(map, LabelType(-1));

  auto make_out = [&map](const auto &labels) {
    LabelType current = 0;
    tf::buffer<LabelType> out;
    out.allocate(labels.size());
    for (auto &&[in, o] : tf::zip(labels, out)) {
      if (map[in] == LabelType(-1))
        map[in] = current++;
      o = map[in];
    }
    return tf::connected_component_labels<LabelType>{std::move(out), current};
  };

  tf::small_vector<tf::connected_component_labels<LabelType>, 4> result;
  result.reserve(per_tag.size());
  for (auto tag_labels : per_tag)
    result.push_back(make_out(tag_labels));
  return result;
}

/// Merge surface and cut component labels into partition_labels for one tag.
/// Cut loops touching uncut surface faces via manifold_edge_link on the
/// original polygon boundary get merged into the same component.
template <typename LabelType, typename Index, typename Policy,
          typename Descriptors, typename Loops, typename Connectivity>
auto merge_labels(tf::connected_component_labels<LabelType> &&sc,
                  tf::connected_component_labels<LabelType> &&cc,
                  const tf::polygons<Policy> &polygons,
                  const Descriptors &descriptors, const Loops &loops,
                  const Connectivity &conn) {
  LabelType offset = sc.n_components;

  tf::buffer<std::array<LabelType, 2>> ids;
  tf::generic_generate(
      tf::zip(cc.labels, descriptors, conn, loops), ids,
      tf::hash_set<std::array<LabelType, 2>, tf::array_hash<LabelType, 2>>{},
      [&, offset](auto tup, auto &buffer, auto &set) {
        auto &&[own_label, d, edge_conn, loop] = tup;
        for (auto &&[nghs, v] : tf::zip(edge_conn, loop)) {
          if (nghs.size() != 0)
            continue;
          if (v.sub_id.label != tf::topo_type::vertex &&
              v.sub_id.label != tf::topo_type::edge)
            continue;
          auto mel = polygons.manifold_edge_link()[d.object][v.sub_id.id];
          if (!mel.is_simple())
            continue;
          auto ngh_poly = mel.face_peer;
          if (sc.labels[ngh_poly] == -1)
            continue;
          std::array<LabelType, 2> pair{sc.labels[ngh_poly],
                                        own_label + offset};
          if (set.insert(pair).second)
            buffer.push_back(pair);
        }
      });

  tbb::parallel_sort(ids.begin(), ids.end());
  ids.erase_till_end(std::unique(ids.begin(), ids.end()));

  tf::buffer<LabelType> map;
  map.allocate(offset + cc.n_components);
  auto n_components = tf::make_dense_equivalence_class_map(ids, map);

  tf::parallel_for_each(
      sc.labels,
      [&map](auto &id) {
        if (id != -1)
          id = map[id];
      },
      tf::checked);
  tf::parallel_for_each(
      cc.labels, [&map, offset](auto &id) { id = map[id + offset]; },
      tf::checked);

  return tf::cut::partition_labels<LabelType>{
      std::move(sc.labels), std::move(cc.labels), n_components};
}

/// Build partition labels for a 2-mesh pair.
template <typename LabelType, typename Index, typename Int, typename Policy0,
          typename Policy1>
auto make_partition_labels(const tf::polygons<Policy0> &form0,
                           const tf::polygons<Policy1> &form1,
                           const tf::face_cuts<Index, Int> &fc,
                           const tf::cut_graph<Index> &cg) {
  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());
  auto conn = cg.connectivity_per_face_edge(fc);
  auto conn_per_tag = tf::make_offset_block_range(fc.tag_offsets(), conn);

  tf::small_vector<tf::connected_component_labels<LabelType>, 4> cut_cls;
  tf::connected_component_labels<LabelType> sc0, sc1;

  tbb::parallel_invoke(
      [&] { cut_cls = make_cut_component_labels<LabelType>(fc, cg); },
      [&] {
        sc0 = make_surface_component_labels2<Index, LabelType>(form0,
                                                              descs_per_tag[0]);
      },
      [&] {
        sc1 = make_surface_component_labels2<Index, LabelType>(form1,
                                                              descs_per_tag[1]);
      });

  tf::small_vector<tf::cut::partition_labels<LabelType>, 4> result;
  result.resize(2);

  tbb::parallel_invoke(
      [&] {
        result[0] = merge_labels<LabelType, Index>(
            std::move(sc0), std::move(cut_cls[0]), form0, descs_per_tag[0],
            loops_per_tag[0], conn_per_tag[0]);
      },
      [&] {
        result[1] = merge_labels<LabelType, Index>(
            std::move(sc1), std::move(cut_cls[1]), form1, descs_per_tag[1],
            loops_per_tag[1], conn_per_tag[1]);
      });

  return result;
}

/// Build partition labels for N meshes.
template <typename LabelType, typename Index, typename Int, typename Iterator,
          std::size_t N>
auto make_partition_labels(tf::range<Iterator, N> forms,
                           const tf::face_cuts<Index, Int> &fc,
                           const tf::cut_graph<Index> &cg) {
  auto n_tags = forms.size();
  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());
  auto conn = cg.connectivity_per_face_edge(fc);
  auto conn_per_tag = tf::make_offset_block_range(fc.tag_offsets(), conn);

  tf::small_vector<tf::connected_component_labels<LabelType>, 4> cut_cls;
  tf::small_vector<tf::connected_component_labels<LabelType>, 4> scs;
  scs.resize(n_tags);

  tbb::task_group tg;
  tg.run([&] { cut_cls = make_cut_component_labels<LabelType>(fc, cg); });
  for (std::size_t t = 0; t < n_tags; ++t)
    tg.run([&, t] {
      scs[t] = make_surface_component_labels2<Index, LabelType>(
          forms[t], descs_per_tag[t]);
    });
  tg.wait();

  tf::small_vector<tf::cut::partition_labels<LabelType>, 4> result;
  result.resize(n_tags);
  for (std::size_t t = 0; t < n_tags; ++t)
    tg.run([&, t] {
      result[t] = merge_labels<LabelType, Index>(
          std::move(scs[t]), std::move(cut_cls[t]), forms[t], descs_per_tag[t],
          loops_per_tag[t], conn_per_tag[t]);
    });
  tg.wait();

  return result;
}

} // namespace tf::cut
