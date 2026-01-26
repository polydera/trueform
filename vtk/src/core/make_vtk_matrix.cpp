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
#include <trueform/vtk/core/make_vtk_matrix.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto make_vtk_matrix(const tf::transformation<float, 3> &transform)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      matrix->SetElement(i, j, transform(i, j));
  return matrix;
}

auto make_vtk_matrix(const tf::transformation<double, 3> &transform)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      matrix->SetElement(i, j, transform(i, j));
  return matrix;
}

auto fill_vtk_matrix(vtkMatrix4x4 *matrix,
                     const tf::transformation<float, 3> &transform) -> void {
  matrix->Identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      matrix->SetElement(i, j, transform(i, j));
  matrix->Modified();
}

auto fill_vtk_matrix(vtkMatrix4x4 *matrix,
                     const tf::transformation<double, 3> &transform) -> void {
  matrix->Identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      matrix->SetElement(i, j, transform(i, j));
  matrix->Modified();
}

} // namespace tf::vtk
