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

#include "../core/point.hpp"
#include <cstddef>

namespace tf::exact {

template <typename IntT, typename RealT, std::size_t Dims>
struct pt_converter_identity {
  template <typename P>
  auto operator()(const P &p) const -> tf::point<IntT, Dims> {
    tf::point<IntT, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<IntT>(p[i]);
    return out;
  }

  auto deconvert(const tf::point<IntT, Dims> &ip) const
      -> tf::point<RealT, Dims> {
    tf::point<RealT, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<RealT>(ip[i]);
    return out;
  }

  auto convert_tolerance(RealT tol) const -> IntT {
    return static_cast<IntT>(tol);
  }

  auto deconvert_tolerance(IntT tol) const -> RealT {
    return static_cast<RealT>(tol);
  }
};

} // namespace tf::exact
