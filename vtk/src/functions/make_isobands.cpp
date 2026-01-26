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
#include <trueform/cut/make_isobands.hpp>
#include <trueform/vtk/functions/make_isobands.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

namespace tf::vtk {

auto make_isobands(vtkPolyData *input, const char *scalars_name,
                   const std::vector<float> &cut_values,
                   const std::vector<int> &selected_bands)
    -> vtkSmartPointer<polydata> {
  if (!input || cut_values.empty() || selected_bands.empty()) {
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
  auto cut_values_r = tf::make_range(cut_values.data(), cut_values.size());
  auto selected_bands_r = tf::make_range(selected_bands.data(), selected_bands.size());

  auto [result_polys, labels] =
      tf::make_isobands<vtkIdType>(polygons, scalars, cut_values_r, selected_bands_r);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(result_polys)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("BandLabel");
  out->GetCellData()->SetScalars(label_array);

  return out;
}

auto make_isobands(vtkPolyData *input, const char *scalars_name,
                   const std::vector<float> &cut_values,
                   const std::vector<int> &selected_bands,
                   tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  if (!input || cut_values.empty() || selected_bands.empty()) {
    return {nullptr, nullptr};
  }

  // Get scalars array
  vtkDataArray *scalars_array = nullptr;
  if (!scalars_name || scalars_name[0] == '\0') {
    scalars_array = input->GetPointData()->GetScalars();
  } else {
    scalars_array = input->GetPointData()->GetArray(scalars_name);
  }

  if (!scalars_array) {
    return {nullptr, nullptr};
  }

  auto *float_array = vtkFloatArray::SafeDownCast(scalars_array);
  if (!float_array) {
    return {nullptr, nullptr};
  }

  auto scalars = tf::make_range(static_cast<float *>(float_array->GetPointer(0)),
                                float_array->GetNumberOfTuples());

  auto polygons = make_polygons(input);
  auto cut_values_r = tf::make_range(cut_values.data(), cut_values.size());
  auto selected_bands_r = tf::make_range(selected_bands.data(), selected_bands.size());

  auto [result_polys, labels, curves] = tf::make_isobands<vtkIdType>(
      polygons, scalars, cut_values_r, selected_bands_r, tf::return_curves);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(result_polys)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("BandLabel");
  out->GetCellData()->SetScalars(label_array);

  auto out_curves = vtkSmartPointer<polydata>::New();
  out_curves->ShallowCopy(make_vtk_polydata(std::move(curves)));

  return {out, out_curves};
}

} // namespace tf::vtk
