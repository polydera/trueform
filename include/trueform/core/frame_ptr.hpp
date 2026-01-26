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

template <typename T, std::size_t Dims>
auto make_frame_ptr(tf::identity_frame<T, Dims> id) {
  return id;
}

template <std::size_t Dims, typename Policy>
auto make_frame_ptr(frame_like<Dims, Policy> &&frame) = delete;
} // namespace tf
