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

template <typename T, std::size_t Dims> class identity_frame {
public:
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  auto transformation() const -> identity_transformation<T, Dims> { return {}; }

  auto inverse_transformation() const -> identity_transformation<T, Dims> {
    return {};
  }
};

} // namespace tf::linalg
