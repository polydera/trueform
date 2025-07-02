/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./frame_like.hpp"
#include "./linalg/frame.hpp"
#include "./linalg/trans_ptr.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1>
using frame_ptr =
    frame_like<Dims, linalg::frame<Dims, linalg::trans_ptr<Dims, Policy0>,
                                   linalg::trans_ptr<Dims, Policy1>>>;

template <std::size_t Dims, typename Policy>
auto make_frame_ptr(const frame_like<Dims, Policy> &frame) {
  return make_frame_like(
      tf::linalg::make_trans_ptr(frame.transformation()),
      tf::linalg::make_trans_ptr(frame.inverse_transformation()));
}

template <std::size_t Dims, typename Policy>
auto make_frame_ptr(frame_like<Dims, Policy> &&frame) = delete;
} // namespace tf
