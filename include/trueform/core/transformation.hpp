/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./linalg/trans.hpp"
#include "./vector_like.hpp"
#include "./coordinate_type.hpp"
#include "./transformation_like.hpp"

namespace tf {
template <typename T, std::size_t Dims>
using transformation =
    tf::transformation_like<Dims, tf::linalg::trans<T, Dims>>;

template <typename T, std::size_t Dims> auto make_identity_transformation() {
  tf::transformation<T, Dims> out;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims + 1; ++j)
      out(i, j) = i == j;
  return out;
}

template <std::size_t Dims, typename T>
auto make_transformation_from_translation(
    const tf::vector_like<Dims, T> &translation) {
  auto out = make_identity_transformation<tf::coordinate_type<T>, Dims>();
  for (std::size_t i = 0; i < Dims; ++i)
    out(i, Dims) = translation[i];
  return out;
}
} // namespace tf
