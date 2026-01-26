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
#include <trueform/vtk/functions/cleaned_polygons.hpp>
#include <trueform/clean.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/make_vtk_data_set_attributes_reindexed.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <vtkCellData.h>
#include <vtkPointData.h>

namespace tf::vtk {

auto cleaned_polygons(polydata *input, float tolerance, bool preserve_data)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  auto [cleaned, face_map, point_map] =
      tf::cleaned<vtkIdType>(input->polygons(), tolerance, tf::return_index_map);

  // Create output
  auto out = vtkSmartPointer<polydata>::New();
  out->Initialize();

  // Set points
  out->SetPoints(make_vtk_points(cleaned.points_buffer()));

  // Set polys
  out->SetPolys(make_vtk_cells(std::move(cleaned.faces_buffer())));

  if (preserve_data) {
    // Remap point data
    auto new_point_data =
        make_vtk_point_data_reindexed(input->GetPointData(), point_map);
    if (new_point_data) {
      out->GetPointData()->ShallowCopy(new_point_data);
    }

    // Remap cell data
    auto new_cell_data =
        make_vtk_cell_data_reindexed(input->GetCellData(), face_map);
    if (new_cell_data) {
      out->GetCellData()->ShallowCopy(new_cell_data);
    }
  }

  return out;
}

} // namespace tf::vtk
