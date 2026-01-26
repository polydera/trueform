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
#include <trueform/core.hpp>
#include <vtkCellArray.h>

namespace tf::vtk {

namespace core {

template <std::size_t V>
struct polys_type {
  using type = decltype(tf::make_blocked_range<V>(
      tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0})));
};

template <>
struct polys_type<tf::dynamic_size> {
  using type = decltype(tf::make_offset_block_range(
      tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0}),
      tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0})));
};

} // namespace core

template <std::size_t V>
using polys_sized_t = typename core::polys_type<V>::type;

using polys_t = polys_sized_t<tf::dynamic_size>;

/// @brief Creates a sized polys view from vtkCellArray (zero-copy).
/// @tparam V Vertex count per polygon (e.g., 3 for triangles).
/// @param cells VTK cell array, may be nullptr.
/// @return A tf::blocked_range<V> view over the connectivity data.
template <std::size_t V>
auto make_polys(vtkCellArray *cells) -> polys_sized_t<V> {
  if (!cells) {
    return tf::make_blocked_range<V>(
        tf::make_range(static_cast<vtkIdType *>(nullptr), 0));
  }
  auto *ptr = static_cast<vtkIdType *>(
      cells->GetConnectivityArray()->GetVoidPointer(0));
  return tf::make_blocked_range<V>(
      tf::make_range(ptr, V * cells->GetNumberOfCells()));
}

/// @brief Creates a polys view from vtkCellArray (zero-copy).
/// @param cells VTK cell array, may be nullptr.
/// @return A tf::offset_block_range view over the connectivity data.
auto make_polys(vtkCellArray *cells) -> polys_t;

template <>
inline auto make_polys<tf::dynamic_size>(vtkCellArray *cells) -> polys_t {
  return make_polys(cells);
}

extern template auto make_polys<3>(vtkCellArray *cells) -> polys_sized_t<3>;

} // namespace tf::vtk
