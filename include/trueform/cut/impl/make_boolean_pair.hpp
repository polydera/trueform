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
#include "../arrangement_class.hpp"
#include "../classify/tagged.hpp"
#include "./ids_common.hpp"
#include "./make_boolean_common.hpp"

namespace tf::cut {

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_boolean_pair(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::arrangement_class, 2> classes) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };
  auto polygons0 = make_polygons(_polygons0);
  auto polygons1 = make_polygons(_polygons1);
  auto [pal0, pal1] = tf::cut::make_classifications<LabelType>(
      polygons0, polygons1, ibp, tcf, classes);
  tf::cut::polygon_arrangement_ids<Index> pai0;
  tf::cut::polygon_arrangement_ids<Index> pai1;
  tbb::parallel_invoke(
      [&pal0 = pal0, &pai0] {
        pai0 = tf::cut::make_polygon_arrangement_ids<Index>(pal0);
      },
      [&pal1 = pal1, &pai1] {
        pai1 = tf::cut::make_polygon_arrangement_ids<Index>(pal1);
      });
  using res_t = decltype(make_boolean_common(
      _polygons0, tf::make_points(ibp.intersection_points()), pai0,
      tcf.descriptors0(), tcf.mapped_loops0(), 0, tf::direction::forward));
  res_t left;
  res_t right;
  auto [direction0, direction1] = tf::make_directions(classes[0], classes[1]);
  tbb::parallel_invoke(
      [&, &direction0 = direction0] {
        left = make_boolean_common(
            _polygons0, tf::make_points(ibp.intersection_points()), pai0,
            tcf.descriptors0(), tcf.mapped_loops0(),
            1, // always use include bucket
            direction0);
      },
      [&, &direction1 = direction1] {
        right = make_boolean_common(
            _polygons1, tf::make_points(ibp.intersection_points()), pai1,
            tcf.descriptors1(), tcf.mapped_loops1(),
            1, // always use include bucket
            direction1);
      });
  return std::make_pair(std::move(left), std::move(right));
}
} // namespace tf::cut
