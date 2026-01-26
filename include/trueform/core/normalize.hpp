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
* Author: Ziga Sajovic
*/
#pragma once
#include "./base/normalize.hpp"
#include "./unit_vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
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
  return core::normalize(v);
}

/// @ingroup core_primitives
/// @brief No-op for unit vectors (already normalized).
template <std::size_t N, typename T>
auto normalize(const unit_vector_like<N, T> &v)
    -> const unit_vector_like<N, T> & {
  return v;
}
} // namespace tf
