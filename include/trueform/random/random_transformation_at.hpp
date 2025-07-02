/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/point_like.hpp"
#include "../core/transformed.hpp"
#include "./random_transformation.hpp"
namespace tf {
template <std::size_t Dims, typename Policy>
auto random_transformation_at(tf::point_like<Dims, Policy> pivot)
    -> tf::transformation<tf::coordinate_type<Policy>, Dims> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation<tf::coordinate_type<Policy>>(
          pivot.as_vector_view()));
}

template <std::size_t Dims, typename Policy, typename Policy1>
auto random_transformation_at(tf::point_like<Dims, Policy> pivot,
                              tf::point_like<Dims, Policy1> new_origin)
    -> tf::transformation<tf::coordinate_type<Policy, Policy1>, Dims> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation<tf::coordinate_type<Policy, Policy1>>(
          new_origin.as_vector_view()));
}
} // namespace tf
