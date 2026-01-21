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
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Tag type representing a principal axis.
///
/// Implicitly converts to @ref tf::unit_vector aligned with axis I.
/// Use with `tf::axis<0>` for X, `tf::axis<1>` for Y, `tf::axis<2>` for Z.
///
/// @tparam I The axis index (0=X, 1=Y, 2=Z, etc.).
template <std::size_t I> struct axis_t {
  template <typename T, std::size_t Dims>
  constexpr operator unit_vector<T, Dims>() const {
    static_assert(I < Dims, "Axis index must be less than dimensions");
    tf::vector<T, Dims> data = tf::zero;
    data[I] = T{1};
    return unit_vector<T, Dims>{tf::unsafe, data};
  }
};

/// @ingroup core_primitives
/// @brief Principal axis instances.
///
/// `tf::axis<0>`, `tf::axis<1>`, `tf::axis<2>` represent X, Y, Z axes.
template <std::size_t I> inline constexpr axis_t<I> axis{};

/// @ingroup core_primitives
/// @brief Create unit vector from axis tag.
/// @overload
template <typename T, std::size_t Dims, std::size_t I>
auto make_unit_vector(tf::axis_t<I> a) -> tf::unit_vector<T, Dims> {
  return a;
}

} // namespace tf
