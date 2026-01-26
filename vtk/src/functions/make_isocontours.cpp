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
#include <trueform/vtk/functions/make_isocontours.hpp>
#include <trueform/intersect/make_isocurves.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

namespace tf::vtk {

auto make_isocontours(vtkPolyData *input, const char *scalars_name,
                      const std::vector<float> &values)
    -> vtkSmartPointer<polydata> {
  if (!input || values.empty()) {
    return nullptr;
  }

  // Get scalars array
  vtkDataArray *scalars_array = nullptr;
  if (!scalars_name || scalars_name[0] == '\0') {
    scalars_array = input->GetPointData()->GetScalars();
  } else {
    scalars_array = input->GetPointData()->GetArray(scalars_name);
  }

  if (!scalars_array) {
    return nullptr;
  }

  auto *float_array = vtkFloatArray::SafeDownCast(scalars_array);
  if (!float_array) {
    return nullptr;
  }

  auto scalars = tf::make_range(static_cast<float *>(float_array->GetPointer(0)),
                                float_array->GetNumberOfTuples());

  auto polygons = make_polygons(input);
  auto cut_values = tf::make_range(values.data(), values.size());

  auto out = vtkSmartPointer<polydata>::New();

  if (values.size() == 1) {
    auto curves = tf::make_isocontours(polygons, scalars, values[0]);
    out->ShallowCopy(make_vtk_polydata(std::move(curves)));
  } else {
    auto curves = tf::make_isocontours(polygons, scalars, cut_values);
    out->ShallowCopy(make_vtk_polydata(std::move(curves)));
  }

  return out;
}

} // namespace tf::vtk
