/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./linalg/trans_view.hpp"
#include "./transformation_like.hpp"

namespace tf {
template <typename T, std::size_t Dims>
using transformation_view =
    tf::transformation_like<Dims, tf::linalg::trans_view<T, Dims>>;

template <std::size_t Dims, typename T> auto make_transformation_view(T *ptr) {
  return transformation_view<T, Dims>{ptr};
}
} // namespace tf
