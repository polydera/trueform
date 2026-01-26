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
#include <trueform/vtk/core/make_vtk_polydata.hpp>

namespace tf::vtk {

auto make_vtk_polydata(
    const tf::polygons_buffer<vtkIdType, float, 3, tf::dynamic_size> &polys)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(polys.points_buffer()));
  out->SetPolys(make_vtk_cells(polys.faces_buffer()));
  return out;
}

auto make_vtk_polydata(
    tf::polygons_buffer<vtkIdType, float, 3, tf::dynamic_size> &&polys)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(std::move(polys.points_buffer())));
  out->SetPolys(make_vtk_cells(std::move(polys.faces_buffer())));
  return out;
}

auto make_vtk_polydata(const tf::curves_buffer<vtkIdType, float, 3> &curves)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(curves.points_buffer()));
  out->SetLines(make_vtk_cells(curves.paths_buffer()));
  return out;
}

auto make_vtk_polydata(tf::curves_buffer<vtkIdType, float, 3> &&curves)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(std::move(curves.points_buffer())));
  out->SetLines(make_vtk_cells(std::move(curves.paths_buffer())));
  return out;
}

auto make_vtk_polydata(const tf::segments_buffer<vtkIdType, float, 3> &segments)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(segments.points_buffer()));
  out->SetLines(make_vtk_cells(segments.edges_buffer()));
  return out;
}

auto make_vtk_polydata(tf::segments_buffer<vtkIdType, float, 3> &&segments)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(std::move(segments.points_buffer())));
  out->SetLines(make_vtk_cells(std::move(segments.edges_buffer())));
  return out;
}

} // namespace tf::vtk
