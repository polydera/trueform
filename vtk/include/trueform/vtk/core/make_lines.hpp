/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <vtkCellArray.h>

namespace tf::vtk {

/// @brief Static lines view type (fixed vertex count V).
template <std::size_t V>
using lines_t = decltype(tf::make_blocked_range<V>(
    tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0})));

/// @brief Dynamic lines view type (variable vertex count).
using dynamic_lines_t = decltype(tf::make_offset_block_range(
    tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0}),
    tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0})));

/// @brief Creates a static lines view from vtkCellArray (zero-copy).
/// @tparam V Vertex count per line (e.g., 2 for segments).
/// @param cells VTK cell array, may be nullptr.
/// @return A tf::blocked_range<V> view over the connectivity data.
template <std::size_t V>
auto make_lines(vtkCellArray *cells) -> lines_t<V> {
  if (!cells) {
    return tf::make_blocked_range<V>(
        tf::make_range(static_cast<vtkIdType *>(nullptr), 0));
  }
  auto *ptr = static_cast<vtkIdType *>(
      cells->GetConnectivityArray()->GetVoidPointer(0));
  return tf::make_blocked_range<V>(
      tf::make_range(ptr, V * cells->GetNumberOfCells()));
}

/// @brief Creates a dynamic lines view from vtkCellArray (zero-copy).
/// @param cells VTK cell array, may be nullptr.
/// @return A tf::offset_block_range view over the connectivity data.
auto make_lines(vtkCellArray *cells) -> dynamic_lines_t;

extern template auto make_lines<2>(vtkCellArray *cells) -> lines_t<2>;

} // namespace tf::vtk
