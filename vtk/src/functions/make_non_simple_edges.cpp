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
#include <trueform/topology/non_simple_edges.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/functions/make_non_simple_edges.hpp>
#include <vtkCellData.h>

namespace tf::vtk {

auto make_non_simple_edges(polydata *input) -> vtkSmartPointer<polydata> {
  if (!input) {
    return nullptr;
  }

  auto [boundary, non_manifold] = tf::make_non_simple_edges(
      input->polygons() | tf::tag(input->face_membership()));

  auto n_boundary = boundary.size();
  auto n_non_manifold = non_manifold.size();
  auto n_total = n_boundary + n_non_manifold;

  // Combine edges into single buffer
  tf::blocked_buffer<vtkIdType, 2> edges;
  edges.allocate(n_total);

  // Create edge type labels: 0=boundary, 1=non-manifold
  tf::buffer<int> labels;
  labels.allocate(n_total);

  auto out_r = tf::zip(edges, labels);

  tf::parallel_copy(tf::zip(boundary, tf::make_constant_range(0, n_boundary)),
                    tf::take(out_r, n_boundary));
  tf::parallel_copy(
      tf::zip(non_manifold, tf::make_constant_range(1, n_non_manifold)),
      tf::drop(out_r, n_boundary));

  auto out = vtkSmartPointer<polydata>::New();
  out->SetPoints(input->GetPoints());
  out->SetLines(make_vtk_cells(std::move(edges)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("EdgeType");
  out->GetCellData()->SetScalars(label_array);

  return out;
}

} // namespace tf::vtk
