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

#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/hash_map.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../arrangement_class.hpp"
#include "../boolean_config.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"
#include "../partition/make_labels.hpp"
#include "./classify_missing.hpp"
#include "./classify_wedge.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {

/// Build per-component classification vote counts for a 2-mesh pair.
///
/// For each intersection edge, classifies other-mesh neighbors using
/// exact wedge predicates. Coplanar pairs are classified directly from
/// cut_graph (aligned/opposing). Missing components (zero votes) are
/// classified via signed distance or ray containment fallback.
///
/// Returns (pal0, pal1, counts0, counts1) where counts[component][i]
/// holds votes: 0=inside, 1=outside, 2=aligned, 3=opposing.
template <typename LabelType, typename Index, typename Policy0, typename Int,
          typename Policy1, typename RealT, std::size_t Dims,
          typename PointsPolicy>
auto make_classification_counts(
    const tf::polygons<Policy0> &polygons0,
    const tf::polygons<Policy1> &polygons1, const tf::face_cuts<Index, Int> &fc,
    const tf::cut_graph<Index> &cg,
    const tf::exact::vertex_converter<Int, RealT, Dims> &conv,
    const tf::points<PointsPolicy> &created_points,
    const tf::boolean_config &config) {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  auto frame0 = tf::frame_of(polygons0);
  auto frame1 = tf::frame_of(polygons1);

  auto get_point = [&](int tag, const vertex_t &v) -> tf::exact::pt3<Int> {
    if (v.source == source::original) {
      if (tag == 0)
        return conv.convert(tf::transformed(polygons0.points()[v.id], frame0));
      else
        return conv.convert(tf::transformed(polygons1.points()[v.id], frame1));
    }
    return created_points[v.id];
  };
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    if (tag == 0)
      f(polygons0.faces()[object]);
    else
      f(polygons1.faces()[object]);
  };
  auto get_created_point = [&](Index id) -> tf::point<RealT, Dims> {
    return conv.deconvert(created_points[id]);
  };
  auto pals = make_partition_labels<LabelType>(polygons0, polygons1, fc, cg);
  auto &pal0 = pals[0];
  auto &pal1 = pals[1];

  tf::blocked_buffer<Index, 4> counts0;
  counts0.allocate(pal0.n_components);
  tf::parallel_fill(counts0.data_buffer(), 0);
  tf::blocked_buffer<Index, 4> counts1;
  counts1.allocate(pal1.n_components);
  tf::parallel_fill(counts1.data_buffer(), 0);

  auto tag_offsets = fc.tag_offsets();
  auto descs = fc.descriptors();
  auto loops = fc.loops();
  auto conn = cg.connectivity_per_face_edge(fc);
  auto coplanar_mask = cg.coplanar_mask();

  auto is_coplanar = [&](auto n_id) { return coplanar_mask[n_id]; };

  auto flat_labels =
      tf::make_mapped_range(tf::make_sequence_range(fc.loops().size()),
                            [&, off = tag_offsets[1]](Index id) {
                              if (id < off)
                                return pal0.cut_labels[id];
                              else
                                return pal1.cut_labels[id - off];
                            });

  // Main wedge classification pass
  using local_map_t =
      std::array<tf::hash_map<LabelType, std::array<Index, 4>>, 2>;
  tf::blocked_reduce(
      tf::enumerate(tf::zip(loops, descs, conn)), std::tie(counts0, counts1),
      local_map_t{},
      [&](const auto &r, auto &lr) {
        for (const auto &[face_id, face_data] : r) {
          const auto &[loop, desc, edge_conn] = face_data;

          if (coplanar_mask[face_id])
            continue;

          auto n = loop.size();
          auto prev = n - 1;
          for (decltype(n) next = 0; next < n; prev = next++) {
            if (!cg.is_intersection_edge(loop[prev], loop[next]))
              continue;

            tf::cut::classify_edge_by_wedge(
                loop, desc, prev, edge_conn[prev], loops, descs, get_point,
                apply_to_face, is_coplanar, flat_labels, lr[1 - desc.tag]);
          }
        }
      },
      [](const auto &lr, auto &result) {
        auto &[res0, res1] = result;
        for (auto [label, cs] : lr[0]) {
          res0[label][0] += cs[0];
          res0[label][1] += cs[1];
          res0[label][2] += cs[2];
          res0[label][3] += cs[3];
        }
        for (auto [label, cs] : lr[1]) {
          res1[label][0] += cs[0];
          res1[label][1] += cs[1];
          res1[label][2] += cs[2];
          res1[label][3] += cs[3];
        }
      });

  // Coplanar pair classification: aligned(2) or opposing(3)
  for (auto &p : cg.coplanar_pairs()) {
    int cls = p.opposing ? 3 : 2;
    auto da = descs[p.loop_a];
    auto db = descs[p.loop_b];
    if (da.tag == 0) {
      auto label = pal0.cut_labels[p.loop_a - tag_offsets[0]];
      if (label >= 0)
        counts0[label][cls]++;
    } else {
      auto label = pal1.cut_labels[p.loop_a - tag_offsets[1]];
      if (label >= 0)
        counts1[label][cls]++;
    }
    if (db.tag == 0) {
      auto label = pal0.cut_labels[p.loop_b - tag_offsets[0]];
      if (label >= 0)
        counts0[label][cls]++;
    } else {
      auto label = pal1.cut_labels[p.loop_b - tag_offsets[1]];
      if (label >= 0)
        counts1[label][cls]++;
    }
  }

  // Classify missing components (no intersection edges)
  classify_missing<Index, LabelType>(
      counts0, counts1, polygons0, polygons1, pal0, pal1, fc, cg,
      config.support_multi_nesting, get_created_point);

  return std::make_tuple(std::move(pal0), std::move(pal1), std::move(counts0),
                         std::move(counts1));
}

/// Build final include/exclude classification labels for a 2-mesh pair.
///
/// Calls make_classification_counts, then remaps each component's
/// label to 0 (exclude) or 1 (include) based on majority vote and
/// the requested arrangement_class flags.
template <typename LabelType, typename Index, typename Policy0,
          typename Policy1, typename RealT, std::size_t Dims, typename Int,
          typename PointsPolicy>
auto make_classification_labels(
    const tf::polygons<Policy0> &polygons0,
    const tf::polygons<Policy1> &polygons1, const tf::face_cuts<Index, Int> &fc,
    const tf::cut_graph<Index> &cg,
    const tf::exact::vertex_converter<Int, RealT, Dims> &conv,
    const tf::points<PointsPolicy> &created_points,
    std::array<tf::arrangement_class, 2> flags,
    const tf::boolean_config &config) {
  auto [pal0, pal1, counts0, counts1] =
      make_classification_counts<LabelType, Index>(
          polygons0, polygons1, fc, cg, conv, created_points, config);

  auto make_classes = [](const auto &counts, auto &pal,
                         tf::arrangement_class flags) {
    pal.n_components = 2;
    auto remap = [&](auto &labels) {
      tf::parallel_for_each(
          labels,
          [&](auto &label) {
            if (label == -1)
              return;
            const auto &count = counts[label];
            int max_id =
                std::max_element(count.begin(), count.end()) - count.begin();
            // 0 = inside, 1 = outside, 2 = aligned, 3 = opposing
            // label 0 = exclude, label 1 = include
            bool include = false;
            switch (max_id) {
            case 0:
              include = flags & tf::arrangement_class::inside;
              break;
            case 1:
              include = flags & tf::arrangement_class::outside;
              break;
            case 2:
              include = flags & tf::arrangement_class::aligned_boundary;
              break;
            case 3:
              include = flags & tf::arrangement_class::opposing_boundary;
              break;
            }
            label = include ? 1 : 0;
          },
          tf::checked);
    };
    tbb::parallel_invoke([&] { remap(pal.polygon_labels); },
                         [&] { remap(pal.cut_labels); });
  };

  tbb::parallel_invoke(
      [&, &pal0 = pal0, &counts0 = counts0] {
        make_classes(counts0, pal0, flags[0]);
      },
      [&, &pal1 = pal1, &counts1 = counts1] {
        make_classes(counts1, pal1, flags[1]);
      });

  tf::small_vector<tf::cut::partition_labels<LabelType>, 4> result;
  result.resize(2);
  result[0] = std::move(pal0);
  result[1] = std::move(pal1);
  return result;
}

} // namespace tf::cut
