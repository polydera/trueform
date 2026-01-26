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
#include "../../core/buffer.hpp"
#include "../../topology/label_connected_components.hpp"
#include "../../topology/make_applier.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../loop/cut_faces.hpp"
#include "../tagged_cut_faces.hpp"
#include "./polygon_arrangement_labels.hpp"

namespace tf::cut {
template <typename Index, typename LabelType, typename Policy, typename Range>
auto make_surface_component_labels(const tf::polygons<Policy> &polygons,
                                   const Range &intersections,
                                   Index expected_components = 2) {
  static_assert(tf::has_manifold_edge_link_policy<Policy>,
                "Use polygons | tf::tag(manifold_edge_link)");
  tf::buffer<char> mask;
  mask.allocate(polygons.faces().size());
  tf::connected_component_labels<LabelType> _components;
  _components.labels.allocate(mask.size());
  tf::parallel_fill(mask, true);
  tf::parallel_for_each(intersections, [&](const auto &r) {
    for (auto i : r) {
      mask[i.object] = false;
      _components.labels[i.object] = -1;
    }
  });
  _components.n_components = tf::label_connected_components_masked(
      _components.labels, mask, tf::make_applier(polygons.manifold_edge_link()),
      expected_components);
  return _components;
}

template <typename LabelType, typename Index, typename ObjectKey>
auto make_cut_face_component_labels(
    const tf::loop::cut_faces<Index, ObjectKey> &cf) {
  tf::connected_component_labels<LabelType> _components;
  _components.labels.allocate(cf.loops().size());
  const auto &conns = cf.connectivity_per_face_edge();
  const auto &loops = cf.mapped_loops();
  _components.n_components = tf::label_connected_components<Index>(
      _components.labels, [&](Index id, const auto &f) {
        const auto &loop = loops[id];
        const auto &conn = conns[id];
        auto size = loop.size();
        auto prev = size - 1;
        for (decltype(size) i = 0; i < size; prev = i++) {
          auto v0 = loop[prev];
          auto v1 = loop[i];
          if (cf.is_intersection_edge(v0, v1) || conn[prev].size() != 1)
            continue;
          f(conn[prev].front());
        }
      });
  return _components;
}

template <typename LabelType, typename Index>
auto make_cut_face_component_labels(const tf::tagged_cut_faces<Index> &cf) {
  auto cl = make_cut_face_component_labels<LabelType>(
      static_cast<const tf::loop::cut_faces<
          Index, tf::loop::tagged_descriptor<Index>> &>(cf));
  auto ls0 = tf::take(cl.labels, cf.partition_id());
  auto ls1 = tf::drop(cl.labels, cf.partition_id());
  tf::buffer<LabelType> map;
  map.allocate(cl.n_components);
  tf::parallel_fill(map, -1);
  auto make_out = [&map](const auto &labels) {
    LabelType current = 0;
    tf::buffer<LabelType> out;
    out.allocate(labels.size());
    for (auto &&[in, out] : tf::zip(labels, out)) {
      if (map[in] == -1)
        map[in] = current++;
      out = map[in];
    }
    return tf::connected_component_labels<LabelType>{std::move(out), current};
  };

  auto cl0 = make_out(ls0);
  auto cl1 = make_out(ls1);
  return std::make_pair(std::move(cl0), std::move(cl1));
}

template <typename LabelType, typename Policy, typename Range0, typename Range1>
auto make_polygon_arrangement_labels(
    tf::connected_component_labels<LabelType> &&sc,
    tf::connected_component_labels<LabelType> &&cc,
    const tf::polygons<Policy> &polygons, const Range0 &zipped_data,
    const Range1 &intersections) {
  LabelType offset = sc.n_components;
  tf::buffer<std::array<LabelType, 2>> ids;
  tf::generic_generate(
      tf::zip(cc.labels, zipped_data), ids,
      tf::hash_set<std::array<LabelType, 2>, tf::array_hash<LabelType, 2>>{},
      [&, offset](auto pair, auto &buffer, auto &set) {
        const auto &[own_label, tup] = pair;
        const auto &[d, conn, loop] = tup;
        for (const auto &[nghs, v] : tf::zip(conn, loop)) {
          if (nghs.size() != 0)
            continue;
          auto i_id = v.intersection_index;
          // when original there is no intersection
          // so we save the index within the face
          int sub_id = i_id;
          if (v.source != tf::loop::vertex_source::original) {
            if (intersections[i_id].target.label == tf::topo_type::face)
              continue;
            else
              sub_id = intersections[i_id].target.id;
          }
          auto mel = polygons.manifold_edge_link()[d.object][sub_id];
          if (!mel.is_simple())
            continue;
          auto ngh_poly = mel.face_peer;
          if (sc.labels[ngh_poly] == -1)
            continue;
          if (set.insert(std::array<LabelType, 2>{sc.labels[ngh_poly],
                                                  own_label + offset})
                  .second)
            buffer.push_back({sc.labels[ngh_poly], own_label + offset});
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
  return tf::cut::polygon_arrangement_labels<LabelType>{
      std::move(sc.labels), std::move(cc.labels), n_components};
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_polygon_arrangement_labels(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf) {
  tf::connected_component_labels<LabelType> cl0;
  tf::connected_component_labels<LabelType> cl1;
  tf::connected_component_labels<LabelType> sl0;
  tf::connected_component_labels<LabelType> sl1;
  tbb::parallel_invoke(
      [&] {
        std::tie(cl0, cl1) =
            tf::cut::make_cut_face_component_labels<LabelType>(tcf);
      },
      [&] {
        sl0 = tf::cut::make_surface_component_labels<Index, LabelType>(
            _polygons0, ibp.intersections0());
      },
      [&] {
        sl1 = tf::cut::make_surface_component_labels<Index, LabelType>(
            _polygons1, ibp.intersections1());
      });
  tf::cut::polygon_arrangement_labels<LabelType> pal0;
  tf::cut::polygon_arrangement_labels<LabelType> pal1;
  tbb::parallel_invoke(
      [&] {
        pal0 = tf::cut::make_polygon_arrangement_labels(
            std::move(sl0), std::move(cl0), _polygons0,
            tf::zip(tcf.descriptors0(), tcf.connectivity_per_face_edge0(),
                    tcf.mapped_loops0()),
            ibp.flat_intersections());
      },
      [&] {
        pal1 = tf::cut::make_polygon_arrangement_labels(
            std::move(sl1), std::move(cl1), _polygons1,
            tf::zip(tcf.descriptors1(), tcf.connectivity_per_face_edge1(),
                    tcf.mapped_loops1()),
            ibp.flat_intersections());
      });
  return std::make_pair(std::move(pal0), std::move(pal1));
}

} // namespace tf::cut
