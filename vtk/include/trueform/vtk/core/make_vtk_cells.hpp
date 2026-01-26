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
#include <vtkIdTypeArray.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Creates vtkCellArray from a blocked_buffer (copies data).
/// @tparam V Vertex count per cell.
/// @param faces Trueform blocked buffer of face indices.
/// @return A new vtkCellArray with copied data.
template <std::size_t V>
auto make_vtk_cells(const tf::blocked_buffer<vtkIdType, V> &faces)
    -> vtkSmartPointer<vtkCellArray> {
  auto cells = vtkSmartPointer<vtkCellArray>::New();
  auto n = faces.size();
  if (n == 0) return cells;

  auto offsets = cells->GetOffsetsArray();
  offsets->SetNumberOfTuples(static_cast<vtkIdType>(n + 1));
  auto *offsets_ptr =
      static_cast<vtkIdType *>(offsets->GetVoidPointer(0));
  tf::parallel_for_each(tf::enumerate(tf::make_range(offsets_ptr, n + 1)),
                     [](auto pair) {
                       auto &&[id, offset] = pair;
                       offset = static_cast<vtkIdType>(V * id);
                     });

  auto conn = cells->GetConnectivityArray();
  conn->SetNumberOfTuples(static_cast<vtkIdType>(V * n));
  auto *conn_ptr = static_cast<vtkIdType *>(conn->GetVoidPointer(0));
  tf::parallel_copy(faces.data_buffer(), tf::make_range(conn_ptr, V * n));

  return cells;
}

/// @brief Creates vtkCellArray from a blocked_buffer (moves data, zero-copy).
/// @tparam V Vertex count per cell.
/// @param faces Trueform blocked buffer with vtkIdType indices (consumed).
/// @return A new vtkCellArray with transferred ownership.
template <std::size_t V>
auto make_vtk_cells(tf::blocked_buffer<vtkIdType, V> &&faces)
    -> vtkSmartPointer<vtkCellArray> {
  auto cells = vtkSmartPointer<vtkCellArray>::New();
  auto n = faces.size();
  if (n == 0) return cells;

  auto offsets = cells->GetOffsetsArray();
  offsets->SetNumberOfTuples(static_cast<vtkIdType>(n + 1));
  auto *offsets_ptr = static_cast<vtkIdType *>(offsets->GetVoidPointer(0));
  tf::parallel_for_each(tf::enumerate(tf::make_range(offsets_ptr, n + 1)),
                     [](auto pair) {
                       auto &&[id, offset] = pair;
                       offset = static_cast<vtkIdType>(V * id);
                     });

  auto *conn_ptr = faces.data_buffer().release();
  auto conn_arr = vtkSmartPointer<vtkIdTypeArray>::New();
  conn_arr->SetNumberOfComponents(1);
  conn_arr->SetArray(conn_ptr, static_cast<vtkIdType>(V * n), 0,
                     vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  cells->SetData(offsets, conn_arr);

  return cells;
}

/// @brief Creates vtkCellArray from an offset_block_buffer (copies data).
/// @param faces Trueform offset block buffer of face indices.
/// @return A new vtkCellArray with copied data.
auto make_vtk_cells(const tf::offset_block_buffer<vtkIdType, vtkIdType> &faces)
    -> vtkSmartPointer<vtkCellArray>;

/// @brief Creates vtkCellArray from an offset_block_buffer (moves data, zero-copy).
/// @param faces Trueform offset block buffer with vtkIdType indices (consumed).
/// @return A new vtkCellArray with transferred ownership.
auto make_vtk_cells(tf::offset_block_buffer<vtkIdType, vtkIdType> &&faces)
    -> vtkSmartPointer<vtkCellArray>;

} // namespace tf::vtk
