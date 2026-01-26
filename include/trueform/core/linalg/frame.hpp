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
#include "../transformation_like.hpp"
namespace tf::linalg {

template <std::size_t Dims, typename Policy0, typename Policy1> class frame {
public:
  using coordinate_type = typename Policy0::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  frame() = default;
  frame(const tf::transformation_like<Dims, Policy0> &_transformation,
        const tf::transformation_like<Dims, Policy1> &_inv_transformation)
      : _transformation{_transformation},
        _inv_transformation{_inv_transformation} {}

  auto transformation() const
      -> const tf::transformation_like<Dims, Policy0> & {
    return _transformation;
  }

  auto inverse_transformation() const
      -> const tf::transformation_like<Dims, Policy1> & {
    return _inv_transformation;
  }

private:
  tf::transformation_like<Dims, Policy0> _transformation;
  tf::transformation_like<Dims, Policy1> _inv_transformation;
};
} // namespace tf::linalg
