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
#include "../core/frame.hpp"
#include "./random_transformation_at.hpp"
namespace tf {
template <std::size_t Dims, typename Policy>
auto random_frame_at(tf::point_like<Dims, Policy> pivot)
    -> tf::frame<tf::coordinate_type<Policy>, Dims> {
  return tf::random_transformation_at<tf::coordinate_type<Policy>>(pivot);
}

template <std::size_t Dims, typename Policy, typename Policy1>
auto random_frame_at(tf::point_like<Dims, Policy> pivot,
                     tf::point_like<Dims, Policy1> new_origin)
    -> tf::frame<tf::coordinate_type<Policy, Policy1>, 3> {
  return tf::random_transformation_at(pivot, new_origin);
}
} // namespace tf
