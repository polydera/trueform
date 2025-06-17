/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./frame_like.hpp"
#include "./inverted.hpp"
#include "./transformation.hpp"

namespace tf {

template <typename RealT, std::size_t Dims>
class frame : public frame_like<Dims, frame<RealT, Dims>> {
public:
  using real_t = RealT;

  frame(const tf::transformation<RealT, Dims> &_transformation)
      : _transformation{_transformation}, _is_dirty{true} {}

  frame(const tf::transformation<RealT, Dims> &_transformation,
        const tf::transformation<RealT, Dims> &_inv_transformation)
      : _transformation{_transformation},
        _inv_transformation{_inv_transformation}, _is_dirty{false} {}

  frame() : _transformation{tf::make_identity_transformation<RealT, Dims>()} {
    _inv_transformation = _transformation;
    _is_dirty = false;
  }

  auto operator=(const tf::transformation<RealT, Dims> &transformation)
      -> frame & {
    _transformation = transformation;
    _is_dirty = true;
    return *this;
  }

  template <typename U> auto fill(const U *ptr) -> void {
    _transformation.fill(ptr);
    _is_dirty = true;
  }

  auto transformation() const -> const tf::transformation<RealT, Dims> & {
    return _transformation;
  }

  auto inverse_transformation() const
      -> const tf::transformation<RealT, Dims> & {
    if (_is_dirty) {
      _inv_transformation = tf::inverted(_transformation);
      _is_dirty = false;
    }
    return _inv_transformation;
  }

private:
  tf::transformation<RealT, Dims> _transformation;
  mutable tf::transformation<RealT, Dims> _inv_transformation;
  mutable bool _is_dirty;
};

template <typename RealT, std::size_t Dims>
auto make_frame(const tf::transformation<RealT, Dims> &_transformation,
                const tf::transformation<RealT, Dims> &_inv_transformation) {
  return tf::frame<RealT, 3>{_transformation, _inv_transformation};
}

template <typename RealT, std::size_t Dims>
auto make_frame(const tf::transformation<RealT, Dims> &_transformation) {
  return tf::frame<RealT, 3>{_transformation};
}
} // namespace tf
