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

#include <array>
#include <cstddef>

namespace tf::cpp::detail {

template <typename Real, std::size_t Dims>
constexpr auto make_identity_transformation_storage()
    -> std::array<Real, Dims *(Dims + 1)> {
  std::array<Real, Dims *(Dims + 1)> values{};
  for (std::size_t row = 0; row != Dims; ++row)
    values[row * (Dims + 1) + row] = Real{1};
  return values;
}

/// A transformation is Dims rows of Dims + 1 columns, so one shared identity
/// per (real, dims) is all a frame slot needs when its owner carries none.
template <typename Real, std::size_t Dims>
inline constexpr std::array<Real, Dims *(Dims + 1)>
    identity_transformation_storage =
        make_identity_transformation_storage<Real, Dims>();

} // namespace tf::cpp::detail
