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
#include <trueform/vtk/core/make_polys.hpp>

namespace tf::vtk {

auto make_polys(vtkCellArray *cells) -> polys_t {
  if (!cells) {
    return tf::make_offset_block_range(
        tf::make_range(static_cast<vtkIdType *>(nullptr), 0),
        tf::make_range(static_cast<vtkIdType *>(nullptr), 0));
  }
  auto *offsets_ptr = static_cast<vtkIdType *>(
      cells->GetOffsetsArray()->GetVoidPointer(0));
  auto *data_ptr = static_cast<vtkIdType *>(
      cells->GetConnectivityArray()->GetVoidPointer(0));
  auto n_cells = cells->GetNumberOfCells();
  auto n_connectivity = cells->GetConnectivityArray()->GetNumberOfValues();
  return tf::make_offset_block_range(
      tf::make_range(offsets_ptr, n_cells + (n_cells != 0)),
      tf::make_range(data_ptr, n_connectivity));
}

template auto make_polys<3>(vtkCellArray *cells) -> polys_sized_t<3>;

} // namespace tf::vtk
