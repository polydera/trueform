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
#include "../core/point_like.hpp"
#include "../core/transformed.hpp"
#include "./random_transformation.hpp"
namespace tf {
template <std::size_t Dims, typename Policy>
auto random_transformation_at(tf::point_like<Dims, Policy> pivot)
    -> tf::transformation<tf::coordinate_type<Policy>, Dims> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation(pivot.as_vector_view()));
}

template <std::size_t Dims, typename Policy, typename Policy1>
auto random_transformation_at(tf::point_like<Dims, Policy> pivot,
                              tf::point_like<Dims, Policy1> new_origin)
    -> tf::transformation<tf::coordinate_type<Policy, Policy1>, Dims> {
  return tf::transformed(
      tf::make_transformation_from_translation(-pivot.as_vector_view()),
      tf::random_transformation(new_origin.as_vector_view()));
}
} // namespace tf
