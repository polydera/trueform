/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/frame.hpp"
#include "./random_transformation.hpp"
namespace tf {

template <std::size_t Dims, typename T>
auto random_frame(tf::vector_like<Dims, T> translation) -> tf::frame<T, Dims> {
  return tf::frame<T, Dims>{tf::random_transformation(translation)};
}

template <typename T, std::size_t Dims>
auto random_frame() -> tf::frame<T, Dims> {
  return tf::frame<T, Dims>{tf::random_transformation<T, Dims>()};
}
} // namespace tf
