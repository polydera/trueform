/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./vector_like.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Normalizes a vector view in place.
///
/// Divides the vector by its Euclidean length, with protection against division
/// by zero. If the length is zero, the vector remains unchanged.
///
/// @tparam T The vector policy
/// @tparam Dims The number of dimensions.
/// @param v A reference to the vector view to normalize.
/// @return Reference to the normalized vector view.
template <std::size_t N, typename T>
auto normalize(vector_like<N, T> &v) -> vector_like<N, T> & {
  auto d = v.length();
  v /= d + (d == 0);
  return v;
}
} // namespace tf
