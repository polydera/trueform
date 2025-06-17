/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./frame.hpp"
#include "./frame_like.hpp"

namespace tf {
template <typename RealT, std::size_t Dims>
class frame_view : public frame_like<Dims, frame_view<RealT, Dims>> {
public:
  using real_t = RealT;

  frame_view(const tf::transformation<RealT, Dims> &_transformation,
             const tf::transformation<RealT, Dims> &_inv_transformation)
      : _transformation{_transformation},
        _inv_transformation{_inv_transformation} {}

  frame_view(const tf::frame<RealT, Dims> &frame)
      : frame_view{frame.transformation(), frame.inverse_transformation()} {}

  auto transformation() const -> const tf::transformation<RealT, Dims> & {
    return _transformation;
  }

  auto inverse_transformation() const
      -> const tf::transformation<RealT, Dims> & {
    return _inv_transformation;
  }

private:
  const tf::transformation<RealT, Dims> &_transformation;
  const tf::transformation<RealT, Dims> &_inv_transformation;
};

template <typename RealT, std::size_t Dims>
auto make_frame_view(const tf::frame<RealT, Dims> &frame)
    -> tf::frame_view<RealT, Dims> {
  return frame_view<RealT, Dims>{frame};
}

} // namespace tf
