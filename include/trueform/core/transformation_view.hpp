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
#include "./linalg/trans_view.hpp"
#include "./transformation_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Non-owning view of transformation matrix data.
///
/// Wraps a raw pointer to interpret it as a transformation matrix.
/// Memory layout: Dims rows of (Dims+1) columns, row-major.
///
/// @tparam T The scalar type.
/// @tparam Dims The coordinate dimensions.
template <typename T, std::size_t Dims>
using transformation_view =
    tf::transformation_like<Dims, tf::linalg::trans_view<T, Dims>>;

/// @ingroup core_primitives
/// @brief Create a transformation view from a pointer.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam T The scalar type.
/// @param ptr Pointer to transformation matrix data.
/// @return A @ref tf::transformation_view wrapping the pointer.
template <std::size_t Dims, typename T> auto make_transformation_view(T *ptr) {
  return transformation_view<T, Dims>{ptr};
}
} // namespace tf
