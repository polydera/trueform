/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./frame.hpp"
#include "./random_transformation_at.hpp"
namespace tf {
template <typename Policy>
auto random_frame_at(tf::point_like<3, Policy> pivot)
    -> tf::frame<tf::value_type<Policy>, 3> {
  return tf::random_transformation_at<tf::value_type<Policy>>(pivot);
}

template <typename Policy, typename Policy1>
auto random_frame_at(tf::point_like<3, Policy> pivot,
                     tf::point_like<3, Policy1> new_origin)
    -> tf::frame<tf::common_value<Policy, Policy1>, 3> {
  return tf::random_transformation_at(pivot, new_origin);
}
} // namespace tf
