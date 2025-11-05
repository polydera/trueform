/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
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
