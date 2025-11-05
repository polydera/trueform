/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./linalg/frame.hpp"
#include "./linalg/identity_frame.hpp"
#include "./linalg/safe_frame.hpp"
#include "./transformation_like.hpp"
#include <utility>
namespace tf {

template <std::size_t Dims, typename Policy> struct frame_like : Policy {
  frame_like() = default;
  frame_like(const Policy &policy) : Policy{policy} {}
  frame_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::inverse_transformation;
  using Policy::Policy;
  using Policy::transformation;

  template <typename T>
  operator tf::frame_like<Dims, tf::linalg::safe_frame<T, Dims>>() const {
    return {transformation(), inverse_transformation()};
  }
};

template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_frame_like(
    const tf::transformation_like<Dims, Policy0> &_transformation,
    const tf::transformation_like<Dims, Policy1> &_inv_transformation) {
  return tf::frame_like<Dims, linalg::frame<Dims, Policy0, Policy1>>{
      _transformation, _inv_transformation};
}

template <typename T, std::size_t Dims>
using identity_frame = frame_like<Dims, linalg::identity_frame<T, Dims>>;

template <typename T0, typename T1, std::size_t Dims>
auto make_frame_like(const identity_transformation<T0, Dims> &,
                     const identity_transformation<T1, Dims> &) {
  return identity_frame<std::common_type_t<T0, T1>, Dims>{};
}
} // namespace tf
