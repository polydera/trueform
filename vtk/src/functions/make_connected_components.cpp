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
#include <trueform/topology/label_connected_components.hpp>
#include <trueform/topology/make_applier.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/functions/make_connected_components.hpp>
#include <vtkCellData.h>

namespace tf::vtk {

auto make_connected_components(polydata *input, tf::connectivity_type type)
    -> std::pair<vtkSmartPointer<polydata>, int> {
  if (!input) {
    return {nullptr, 0};
  }

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(input);

  tf::buffer<vtkIdType> labels;
  labels.allocate(input->GetNumberOfPolys());

  int n_components = 0;

  switch (type) {
  case tf::connectivity_type::manifold_edge:
    n_components = tf::label_connected_components<vtkIdType>(
        labels, tf::make_applier(input->manifold_edge_link()));
    break;
  case tf::connectivity_type::edge:
    n_components = tf::label_connected_components<vtkIdType>(
        labels, tf::make_applier(input->face_link()));
    break;
  case tf::connectivity_type::vertex: {
    // vertex_link connects vertices, so label vertices first
    tf::buffer<vtkIdType> vertex_labels;
    vertex_labels.allocate(input->GetNumberOfPoints());
    n_components = tf::label_connected_components<vtkIdType>(
        vertex_labels, tf::make_applier(input->vertex_link()));

    // Transfer vertex labels to faces (use label of first vertex)
    auto polys = input->polys();
    tf::parallel_transform(polys, labels, [&vertex_labels](const auto &face) {
      return vertex_labels[face[0]];
    });
    break;
  }
  }

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("ComponentLabel");
  out->GetCellData()->SetScalars(label_array);

  return {out, n_components};
}

} // namespace tf::vtk
