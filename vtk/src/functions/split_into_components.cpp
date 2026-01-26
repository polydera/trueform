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
#include <trueform/vtk/functions/split_into_components.hpp>
#include <trueform/reindex/split_into_components.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <trueform/vtk/core/make_range.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <vtkCellData.h>
#include <vtkIdTypeArray.h>

namespace tf::vtk {

auto split_into_components(vtkPolyData *input, const char *array_name)
    -> std::pair<std::vector<vtkSmartPointer<polydata>>, std::vector<vtkIdType>> {
  if (!input) {
    return {};
  }

  auto *label_array =
      vtkIdTypeArray::SafeDownCast(input->GetCellData()->GetArray(array_name));
  if (!label_array) {
    return {};
  }

  auto labels = make_range(label_array);
  auto polygons = make_polygons(input);

  auto [components, component_labels] =
      tf::split_into_components(polygons, labels);

  std::vector<vtkSmartPointer<polydata>> result;
  result.reserve(components.size());

  for (auto &component : components) {
    auto vtk_poly = make_vtk_polydata(std::move(component));
    auto pd = vtkSmartPointer<polydata>::New();
    pd->ShallowCopy(vtk_poly);
    result.push_back(pd);
  }

  return {std::move(result), std::move(component_labels)};
}

auto split_into_components(vtkPolyData *input)
    -> std::pair<std::vector<vtkSmartPointer<polydata>>, std::vector<vtkIdType>> {
  if (!input || !input->GetCellData()->GetScalars()) {
    return {};
  }
  return split_into_components(input, input->GetCellData()->GetScalars()->GetName());
}

} // namespace tf::vtk
