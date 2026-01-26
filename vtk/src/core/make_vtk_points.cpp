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
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <vtkFloatArray.h>
#include <vtkPoints.h>

namespace tf::vtk {

auto make_vtk_points(const tf::points_buffer<float, 3> &points)
    -> vtkSmartPointer<vtkPoints> {
  auto vtk_points = vtkSmartPointer<vtkPoints>::New();
  vtk_points->SetNumberOfPoints(static_cast<vtkIdType>(points.size()));
  tf::parallel_copy(points.points(), make_points(vtk_points.Get()));
  return vtk_points;
}

auto make_vtk_points(tf::points_buffer<float, 3> &&points)
    -> vtkSmartPointer<vtkPoints> {
  auto n = points.size();
  auto *ptr = points.data_buffer().release();

  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(3);
  arr->SetArray(ptr, static_cast<vtkIdType>(3 * n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);

  auto vtk_points = vtkSmartPointer<vtkPoints>::New();
  vtk_points->SetData(arr);
  return vtk_points;
}

} // namespace tf::vtk
