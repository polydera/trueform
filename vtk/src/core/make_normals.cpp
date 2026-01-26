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
#include <trueform/vtk/core/make_normals.hpp>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>

namespace tf::vtk {

auto make_point_normals(vtkPolyData *poly) -> normals_t {
  if (!poly) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *point_data = poly->GetPointData();
  if (!point_data) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *normals = point_data->GetNormals();
  if (!normals) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *ptr = static_cast<float *>(normals->GetVoidPointer(0));
  return tf::make_unit_vectors<3>(
      tf::make_range(ptr, 3 * normals->GetNumberOfTuples()));
}

auto make_cell_normals(vtkPolyData *poly) -> normals_t {
  if (!poly) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *cell_data = poly->GetCellData();
  if (!cell_data) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *normals = cell_data->GetNormals();
  if (!normals) {
    return tf::make_unit_vectors<3>(
        tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *ptr = static_cast<float *>(normals->GetVoidPointer(0));
  return tf::make_unit_vectors<3>(
      tf::make_range(ptr, 3 * normals->GetNumberOfTuples()));
}

} // namespace tf::vtk
