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
