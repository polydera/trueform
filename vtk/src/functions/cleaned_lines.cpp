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
#include <trueform/vtk/functions/cleaned_lines.hpp>
#include <trueform/clean.hpp>
#include <trueform/vtk/core/make_vtk_data_set_attributes_reindexed.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <vtkPointData.h>

namespace tf::vtk {

auto cleaned_lines(polydata *input, float tolerance, bool preserve_data)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfLines() == 0) {
    return nullptr;
  }

  // Clean curves (merge points, remove degenerate edges, reconnect paths)
  auto [cleaned, point_map] =
      tf::cleaned<vtkIdType>(input->curves(), tolerance, tf::return_index_map);

  // Create output
  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(cleaned)));

  if (preserve_data) {
    // Remap point data
    auto new_point_data =
        make_vtk_point_data_reindexed(input->GetPointData(), point_map);
    if (new_point_data) {
      out->GetPointData()->ShallowCopy(new_point_data);
    }

    // Note: cell data cannot be remapped because edges are reconnected into paths
    // (original cell count != output cell count)
  }

  return out;
}

} // namespace tf::vtk
