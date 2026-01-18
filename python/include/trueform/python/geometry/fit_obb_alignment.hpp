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

#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/geometry/fit_obb_alignment.hpp>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/core/form.hpp>
#include <trueform/spatial/policy/tree.hpp>

namespace tf::py {

template <typename RealT, std::size_t Dims>
auto fit_obb_alignment_impl(point_cloud_wrapper<RealT, Dims> &cloud0,
                            point_cloud_wrapper<RealT, Dims> &cloud1,
                            std::size_t sample_size) {
  auto pts0 = cloud0.make_primitive_range();
  auto pts1 = cloud1.make_primitive_range();

  bool has0 = cloud0.has_transformation();
  bool has1 = cloud1.has_transformation();

  // tree() auto-builds if needed
  auto form1 = pts1 | tf::tag(cloud1.tree());

  auto compute = [&]() {
    if (has0 && has1) {
      return tf::fit_obb_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())),
          form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          sample_size);
    } else if (has0) {
      return tf::fit_obb_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())), form1,
          sample_size);
    } else if (has1) {
      return tf::fit_obb_alignment(
          pts0, form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          sample_size);
    } else {
      return tf::fit_obb_alignment(pts0, form1, sample_size);
    }
  };

  return make_numpy_array(compute());
}

} // namespace tf::py
