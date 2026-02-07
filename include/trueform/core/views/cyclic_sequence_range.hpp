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

#include "./mapped_range.hpp"
#include "./sequence_range.hpp"

#include <algorithm>

namespace tf {

/// @ingroup core_ranges
/// @brief Creates a cyclic sequence with explicit stride and start.
///
/// Produces `size` indices using formula: `(start + i * stride) % modulo`.
/// Useful for strided sampling with wrap-around.
///
/// @param size Number of indices to produce.
/// @param modulo The wrap-around boundary.
/// @param start Starting index.
/// @param stride Step between consecutive indices.
/// @return A lazy view of cyclic indices.
///
/// @code
/// // 100 indices with stride 3, starting at 5, wrapping at 1000
/// auto ids = make_cyclic_sequence_range(100, 1000, 5, 3);
/// // Produces: 5, 8, 11, 14, ..., (5 + 99*3) % 1000
/// @endcode
template <typename T>
auto make_cyclic_sequence_range(T size, T modulo, T start, T stride) {
  static_assert(std::is_integral_v<T>, "T must be integral.");
  return tf::make_mapped_range(tf::make_sequence_range(size),
                               [modulo, start, stride](auto i) {
                                 return (start + i * stride) % modulo;
                               });
}

/// @ingroup core_ranges
/// @brief Creates a cyclic sequence with computed stride.
///
/// Produces `size` indices with stride computed as `max(1, modulo / size)`.
/// If size >= modulo, stride is 1 (cycling through indices).
///
/// @param size Number of indices to produce.
/// @param modulo The wrap-around boundary.
/// @param start Starting index.
/// @return A lazy view of cyclic indices.
///
/// @code
/// // 100 evenly spaced indices starting at 5, wrapping at 1000
/// auto ids = make_cyclic_sequence_range(100, 1000, 5);  // stride = 10
/// @endcode
template <typename T>
auto make_cyclic_sequence_range(T size, T modulo, T start) {
  static_assert(std::is_integral_v<T>, "T must be integral.");
  T stride = std::max(T(1), modulo / size);
  return make_cyclic_sequence_range(size, modulo, start, stride);
}

/// @ingroup core_ranges
/// @brief Creates a cyclic sequence with computed stride, starting at 0.
///
/// Produces `size` indices with stride computed as `max(1, modulo / size)`,
/// starting from index 0.
///
/// @param size Number of indices to produce.
/// @param modulo The wrap-around boundary.
/// @return A lazy view of cyclic indices.
///
/// @code
/// // 100 evenly spaced indices from range [0, 1000)
/// auto ids = make_cyclic_sequence_range(100, 1000);  // stride = 10
/// // Produces: 0, 10, 20, 30, ...
/// @endcode
template <typename T>
auto make_cyclic_sequence_range(T size, T modulo) {
  static_assert(std::is_integral_v<T>, "T must be integral.");
  return make_cyclic_sequence_range(size, modulo, T(0));
}

} // namespace tf
