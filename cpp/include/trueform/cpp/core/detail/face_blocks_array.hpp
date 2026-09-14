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

#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>

namespace tf::cpp::detail {

/// @brief The array a per-face-side fact is stated in, at the arity that
/// states it.
///
/// A triangle mesh names three consecutive sides per face, so one `[N, 3]`
/// block states the whole of it; a mixed one carries its own offsets. It is
/// the one producer of that shape, so a caller can hand back what an entry
/// produced and a cache can take what an entry returns.
template <typename Index, std::size_t Ngon> struct face_blocks_array;

template <typename Index> struct face_blocks_array<Index, 3> {
  using type = nd_array<Index>;
};

template <typename Index> struct face_blocks_array<Index, tf::dynamic_size> {
  using type = offset_blocked_buffer<Index, Index>;
};

template <typename Index, std::size_t Ngon>
using face_blocks_array_t = typename face_blocks_array<Index, Ngon>::type;

} // namespace tf::cpp::detail
