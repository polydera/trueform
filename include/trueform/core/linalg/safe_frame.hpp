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
#include "../inverted.hpp"
#include "../transformation.hpp"
namespace tf::linalg {

template <typename RealT, std::size_t Dims> class safe_frame {
public:
  using coordinate_type = RealT;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  safe_frame(const tf::transformation<RealT, Dims> &_transformation)
      : _transformation{_transformation}, _is_dirty{true} {}

  safe_frame(const tf::transformation<RealT, Dims> &_transformation,
             const tf::transformation<RealT, Dims> &_inv_transformation)
      : _transformation{_transformation},
        _inv_transformation{_inv_transformation}, _is_dirty{false} {}

  safe_frame()
      : _transformation{tf::make_identity_transformation<RealT, Dims>()} {
    _inv_transformation = _transformation;
    _is_dirty = false;
  }

  auto operator=(const tf::transformation<RealT, Dims> &transformation)
      -> safe_frame & {
    _transformation = transformation;
    _is_dirty = true;
    return *this;
  }

  auto set(const tf::transformation<RealT, Dims> &transformation,
           const tf::transformation<RealT, Dims> &inv_transformation) {
    _transformation = transformation;
    _inv_transformation = inv_transformation;
    _is_dirty = false;
  }

  template <typename U> auto fill(const U *ptr) -> void {
    _transformation.fill(ptr);
    _is_dirty = true;
  }

  template <typename U> auto fill(const U *ptr, const U *inv_ptr) -> void {
    _transformation.fill(ptr);
    _inv_transformation.fill(inv_ptr);
    _is_dirty = false;
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

} // namespace tf::linalg
