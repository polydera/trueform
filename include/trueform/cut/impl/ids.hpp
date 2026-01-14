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
#include "./ids_common.hpp"
#include "./labels.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {
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
