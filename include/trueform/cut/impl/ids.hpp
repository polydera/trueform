/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "./labels.hpp"
#include "./polygon_arrangement_ids.hpp"
#include "./polygon_arrangement_labels.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"

namespace tf::cut {
template <typename Index, typename LabelType>
auto make_polygon_arrangement_ids(
    const tf::cut::polygon_arrangement_labels<LabelType> &labels) {
  tf::cut::polygon_arrangement_ids<Index> pai;
  auto make_f = [&, n_components = labels.n_components](
                    tf::offset_block_buffer<Index, Index> &b,
                    const auto &labels) {
    b.offsets_buffer().allocate(n_components + 1);
    b.data_buffer().allocate(labels.size());
    tf::parallel_iota(b.data_buffer(), 0);
    tbb::parallel_sort(b.data_buffer().begin(), b.data_buffer().end(),
                       [&](auto i0, auto i1) {
                         return std::make_pair(labels[i0] == -1, labels[i0]) <
                                std::make_pair(labels[i1] == -1, labels[i1]);
                       });
    auto oi = b.offsets_buffer().begin();
    auto oi_end = b.offsets_buffer().end();
    *oi++ = 0;
    Index current = 0;
    auto it = b.data_buffer().begin();
    auto end = b.data_buffer().end();
    auto begin = b.data_buffer().begin();
    while (oi != oi_end) {
      auto next =
          std::find_if(it, end, [&](auto i) { return labels[i] != current; });
      *oi++ = next - begin;
      current++;
      it = next;
    }
  };
  tbb::parallel_invoke([&] { make_f(pai.polygons, labels.polygon_labels); },
                       [&] { make_f(pai.cut_faces, labels.cut_labels); });
  return pai;
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_polygon_arrangement_ids(
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
  tf::cut::polygon_arrangement_ids<Index> pai0;
  tf::cut::polygon_arrangement_ids<Index> pai1;
  tbb::parallel_invoke(
      [&] {
        auto pal0 = tf::cut::make_polygon_arrangement_labels(
            std::move(sl0), std::move(cl0), _polygons0,
            tf::zip(tcf.descriptors0(), tcf.connectivity_per_face_edge0(),
                    tcf.mapped_loops0()),
            ibp.flat_intersections());
        pai0 = tf::cut::make_polygon_arrangement_ids<Index>(pal0);
      },
      [&] {
        auto pal1 = tf::cut::make_polygon_arrangement_labels(
            std::move(sl1), std::move(cl1), _polygons1,
            tf::zip(tcf.descriptors1(), tcf.connectivity_per_face_edge1(),
                    tcf.mapped_loops1()),
            ibp.flat_intersections());
        pai1 = tf::cut::make_polygon_arrangement_ids<Index>(pal1);
      });
  return std::make_pair(std::move(pai0), std::move(pai1));
}

} // namespace tf::cut
