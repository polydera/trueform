/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./point_like.hpp"
#include "./random_transformation.hpp"
#include "./transformed.hpp"
namespace tf {
template <typename Policy>
auto random_transformation_at(tf::point_like<3, Policy> pivot)
    -> tf::transformation<tf::value_type<Policy>, 3> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation<tf::value_type<Policy>>(
          pivot.as_vector_view()));
}

template <typename Policy, typename Policy1>
auto random_transformation_at(tf::point_like<3, Policy> pivot,
                              tf::point_like<3, Policy1> new_origin)
    -> tf::transformation<tf::common_value<Policy, Policy1>, 3> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation<tf::common_value<Policy, Policy1>>(
          new_origin.as_vector_view()));
}
} // namespace tf
