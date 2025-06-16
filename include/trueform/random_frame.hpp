/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./frame.hpp"
#include "./random_transformation.hpp"
namespace tf {

template <typename T>
auto random_frame(tf::vector<T, 3> translation = {{0, 0, 0}})
    -> tf::frame<T, 3> {
  return tf::frame<T, 3>{tf::random_transformation(translation)};
}
} // namespace tf
