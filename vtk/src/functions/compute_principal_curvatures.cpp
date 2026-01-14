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
 * Author: Ziga Sajovic
 */
#include <trueform/geometry.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/compute_principal_curvatures.hpp>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

namespace tf::vtk {

auto compute_principal_curvatures(polydata *input, int k,
                                  bool compute_directions) -> void {
  if (!input) {
    return;
  }

  auto polygons = input->polygons() | tf::tag(input->face_membership()) |
                  tf::tag(input->vertex_link());
  const auto n_points = static_cast<std::size_t>(input->GetNumberOfPoints());

  tf::buffer<float> k1, k2;
  k1.allocate(n_points);
  k2.allocate(n_points);

  if (compute_directions) {
    tf::unit_vectors_buffer<float, 3> d1, d2;
    d1.allocate(n_points);
    d2.allocate(n_points);

    tf::compute_principal_curvatures(polygons, k1, k2, d1, d2,
                                     static_cast<std::size_t>(k));

    auto d1_array = make_vtk_array(std::move(d1));
    d1_array->SetName("D1");
    input->GetPointData()->AddArray(d1_array);

    auto d2_array = make_vtk_array(std::move(d2));
    d2_array->SetName("D2");
    input->GetPointData()->AddArray(d2_array);
  } else {
    tf::compute_principal_curvatures(polygons, k1, k2,
                                     static_cast<std::size_t>(k));
  }

  auto k1_array = make_vtk_array(std::move(k1));
  k1_array->SetName("K1");
  input->GetPointData()->AddArray(k1_array);

  auto k2_array = make_vtk_array(std::move(k2));
  k2_array->SetName("K2");
  input->GetPointData()->AddArray(k2_array);
}

} // namespace tf::vtk
