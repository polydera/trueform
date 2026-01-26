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
#include "../../core/blocked_buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/views/zip.hpp"
#include "../../spatial/classify/point_in_mesh.hpp"
#include "../../topology/set_component_labels.hpp"
#include "../arrangement_class.hpp"
#include "../impl/labels.hpp"
#include "../tagged_cut_faces.hpp"
#include "./classify_on_shared_edge.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"

namespace tf::cut {
template <typename Policy, typename Range0, typename Range1, typename LabelType>
auto compute_joint_components(
    const tf::polygons<Policy> &polygons, const Range0 &zipped,
    const Range1 &descriptors,
    const tf::cut::polygon_arrangement_labels<LabelType> &pal,
    std::size_t offset) {
  tf::buffer<std::array<LabelType, 2>> ids;
  tf::buffer<LabelType> labels;
  labels.allocate(pal.polygon_labels.size());
  tf::generic_generate(
      zipped, ids,
      tf::hash_set<std::array<LabelType, 2>, tf::array_hash<LabelType, 2>>{},
      [&](const auto &pair, auto &ids, auto &set) {
        const auto &[d, label, conn] = pair;
        labels[d.object] = label;
        for (auto nexts : conn) {
          // we only consider manifold edge connected components
          if (std::count_if(nexts.begin(), nexts.end(),
                            [&d = d, &descriptors](auto x) {
                              return descriptors[x].tag == d.tag;
                            }) != 1)
            continue;
          for (auto next : nexts) {
            if (d.tag != descriptors[next].tag)
              continue;
            std::array<LabelType, 2> _ids{pal.cut_labels[next - offset], label};
            if (_ids[0] == _ids[1])
              continue;
            if (_ids[0] > _ids[1])
              std::swap(_ids[0], _ids[1]);
            if (set.insert(_ids).second)
              ids.push_back(_ids);
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

template <typename Policy0, typename Policy1, typename Index, typename RealT,
          std::size_t Dims, typename LabelType>
auto classify_missing_components(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::blocked_buffer<Index, 4> &counts0,
    tf::blocked_buffer<Index, 4> &counts1,
    const tf::cut::polygon_arrangement_labels<LabelType> &pal0,
    const tf::cut::polygon_arrangement_labels<LabelType> &pal1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf) {
  auto any_missing = [](const auto &counts) {
    Index count = 0;
    for (auto c : counts)
      count += c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0;
    return count;
  };

  // we collect  representative for each component
  // technically this is a race condition. But we
  // do not care which id gets written,
  // and parallel_apply end synchronizes
  auto make_reprs = [](const auto &labels, auto n_components) {
    tf::buffer<Index> reprs;
    reprs.allocate(n_components);
    tf::parallel_fill(reprs, -1);
    tf::parallel_for_each(tf::enumerate(labels), [&](auto pair) {
      const auto &[id, label] = pair;
      if (label == -1)
        return;
      reprs[label] = id;
    });
    return reprs;
  };

  auto compute_missing = [&tcf, &ibp, &any_missing, &make_reprs](
                             auto &counts, const auto &zipped_other,
                             const auto &polygons, const auto &polygons_other,
                             const auto &pal, const auto &pal_other,
                             const auto &loops, std::size_t offset) {
    if (!any_missing(counts))
      return;
    auto jlabels_other = compute_joint_components(
        polygons_other, zipped_other, tcf.descriptors(), pal_other, offset);
    auto reprs_poly = make_reprs(pal.polygon_labels, pal.n_components);
    auto reprs_cut = make_reprs(pal.cut_labels, pal.n_components);
    auto frame = tf::frame_of(polygons);
    auto get_loop_point = [&](auto v) {
      tf::point<tf::coordinate_type<Policy1>, tf::coordinate_dims_v<Policy1>>
          pt;
      if (v.source == tf::loop::vertex_source::original)
        pt = tf::transformed(polygons.points()[v.id], frame);
      else
        pt = ibp.intersection_points()[v.id];
      return pt;
    };
    tf::parallel_for_each(
        tf::zip(counts, reprs_poly, reprs_cut), [&](auto tup) {
          auto &&[count, poly_id, cut_id] = tup;
          if (!(count[0] == 0 && count[1] == 0 && count[2] == 0 &&
                count[3] == 0))
            return;
          else if (poly_id != -1) {
            auto point =
                tf::transformed(tf::centroid(polygons[poly_id]), frame);
            auto c = tf::spatial::classify_point(point, polygons_other,
                                                 jlabels_other);
            count[c == tf::containment::outside] = 1;
          } else {
            auto point = tf::centroid(tf::make_polygon(
                tf::make_mapped_range(loops[cut_id], get_loop_point)));
            auto c = tf::spatial::classify_point(point, polygons_other,
                                                 jlabels_other);
            count[c == tf::containment::outside] = 1;
          }
        });
  };

  tbb::parallel_invoke(
      [&] {
        compute_missing(counts1,
                        tf::zip(tcf.descriptors0(), pal0.cut_labels,
                                tcf.connectivity_per_face_edge0()),
                        _polygons1, _polygons0, pal1, pal0, tcf.mapped_loops1(),
                        0);
      },
      [&] {
        compute_missing(counts0,
                        tf::zip(tcf.descriptors1(), pal1.cut_labels,
                                tcf.connectivity_per_face_edge1()),
                        _polygons0, _polygons1, pal0, pal1, tcf.mapped_loops0(),
                        tcf.partition_id());
      });
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_classification_counts(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf) {
  auto [pal0, pal1] = make_polygon_arrangement_labels<LabelType>(
      _polygons0, _polygons1, ibp, tcf);
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };

  auto polygons0 = make_polygons(_polygons0);
  auto polygons1 = make_polygons(_polygons1);

  tf::blocked_buffer<Index, 4> counts0;
  counts0.allocate(pal0.n_components);
  tf::parallel_fill(counts0.data_buffer(), 0);
  tf::blocked_buffer<Index, 4> counts1;
  counts1.allocate(pal1.n_components);
  tf::parallel_fill(counts1.data_buffer(), 0);

  auto zipped = tf::zip(tcf.loops(), tcf.mapped_loops(), tcf.descriptors());
  tf::blocked_reduce(
      tf::enumerate(zipped), std::tie(counts0, counts1),
      std::array<tf::hash_map<LabelType, std::array<Index, 4>>, 2>{},
      [&tcf, &polygons0, &polygons1, &ibp, &zipped, &pal0 = pal0,
       &pal1 = pal1](const auto &r, auto &local_r) {
        for (const auto &[face_id, face_data] : r) {
          const auto &[loop0, mapped_loop0, d0] = face_data;

          auto process_face =
              [&, &loop0 = loop0, &mapped_loop0 = mapped_loop0,
               &d0 = d0](const auto &polys_self, const auto &polys_other,
                         const auto &pal_other, Index partition_other,
                         const auto &conn) {
                auto size = loop0.size();
                auto curr = size - 1;
                for (decltype(size) next = 0; next < size; curr = next++) {
                  auto v0 = mapped_loop0[curr];
                  auto v1 = mapped_loop0[next];

                  if (!tcf.is_intersection_edge(v0, v1))
                    continue;

                  tf::cut::classify_by_wedge_on_shared_edge(
                      std::forward_as_tuple(loop0, mapped_loop0, d0), curr,
                      conn[curr], zipped, polys_self, polys_other, pal_other,
                      partition_other, local_r,
                      tf::make_points(ibp.intersection_points()));
                }
              };

          if (d0.tag == 0) {
            const auto &conn = tcf.connectivity_per_face_edge0()[face_id];
            process_face(polygons0, polygons1, pal1, tcf.partition_id(), conn);
          } else {
            const auto &conn =
                tcf.connectivity_per_face_edge1()[face_id - tcf.partition_id()];
            process_face(polygons1, polygons0, pal0, Index{0}, conn);
          }
        }
      },
      [](const auto &local, auto &result) {
        auto &[res0, res1] = result;
        for (auto [label, cs] : local[0]) {
          res0[label][0] += cs[0];
          res0[label][1] += cs[1];
          res0[label][2] += cs[2];
          res0[label][3] += cs[3];
        }
        for (auto [label, cs] : local[1]) {
          res1[label][0] += cs[0];
          res1[label][1] += cs[1];
          res1[label][2] += cs[2];
          res1[label][3] += cs[3];
        }
      });

  classify_missing_components(polygons0, polygons1, counts0, counts1, pal0,
                              pal1, ibp, tcf);
  return std::make_tuple(std::move(pal0), std::move(pal1), std::move(counts0),
                         std::move(counts1));
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_classifications(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::arrangement_class, 2> flags) {
  auto [pal0, pal1, counts0, counts1] =
      make_classification_counts<LabelType>(_polygons0, _polygons1, ibp, tcf);

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
  return std::make_pair(std::move(pal0), std::move(pal1));
}

} // namespace tf::cut
