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
#include <trueform/vtk/core/make_vtk_cells.hpp>

namespace tf::vtk {

auto make_vtk_cells(const tf::offset_block_buffer<vtkIdType, vtkIdType> &faces)
    -> vtkSmartPointer<vtkCellArray> {
  auto cells = vtkSmartPointer<vtkCellArray>::New();
  auto n = faces.size();
  if (n == 0) return cells;

  auto offsets = cells->GetOffsetsArray();
  offsets->SetNumberOfTuples(static_cast<vtkIdType>(n + (n != 0)));
  auto *offsets_ptr = static_cast<vtkIdType *>(offsets->GetVoidPointer(0));
  tf::parallel_copy(faces.offsets_buffer(),
                    tf::make_range(offsets_ptr, n + (n != 0)));

  auto conn = cells->GetConnectivityArray();
  auto n_conn = faces.data_buffer().size();
  conn->SetNumberOfTuples(static_cast<vtkIdType>(n_conn));
  auto *conn_ptr = static_cast<vtkIdType *>(conn->GetVoidPointer(0));
  tf::parallel_copy(faces.data_buffer(), tf::make_range(conn_ptr, n_conn));

  return cells;
}

auto make_vtk_cells(tf::offset_block_buffer<vtkIdType, vtkIdType> &&faces)
    -> vtkSmartPointer<vtkCellArray> {
  auto cells = vtkSmartPointer<vtkCellArray>::New();
  auto n = faces.size();
  if (n == 0) return cells;

  auto n_offsets = n + (n != 0);
  auto n_conn = faces.data_buffer().size();
  auto *offsets_ptr = faces.offsets_buffer().release();
  auto *conn_ptr = faces.data_buffer().release();

  auto offsets_arr = vtkSmartPointer<vtkIdTypeArray>::New();
  offsets_arr->SetNumberOfComponents(1);
  offsets_arr->SetArray(offsets_ptr, static_cast<vtkIdType>(n_offsets), 0,
                        vtkAbstractArray::VTK_DATA_ARRAY_DELETE);

  auto conn_arr = vtkSmartPointer<vtkIdTypeArray>::New();
  conn_arr->SetNumberOfComponents(1);
  conn_arr->SetArray(conn_ptr, static_cast<vtkIdType>(n_conn), 0,
                     vtkAbstractArray::VTK_DATA_ARRAY_DELETE);

  cells->SetData(offsets_arr, conn_arr);
  return cells;
}

} // namespace tf::vtk
