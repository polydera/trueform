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
#include <trueform/vtk/core/make_vtk_array_reindexed.hpp>
#include <trueform/vtk/core/make_vtk_data_set_attributes_reindexed.hpp>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>

namespace tf::vtk {

auto make_vtk_data_set_attributes_reindexed(
    vtkDataSetAttributes *attr, const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkDataSetAttributes> {
  if (!attr) {
    return nullptr;
  }

  auto out = vtkSmartPointer<vtkDataSetAttributes>::Take(attr->NewInstance());

  // Store active attribute names
  const char *scalars_name =
      attr->GetScalars() ? attr->GetScalars()->GetName() : nullptr;
  const char *vectors_name =
      attr->GetVectors() ? attr->GetVectors()->GetName() : nullptr;
  const char *normals_name =
      attr->GetNormals() ? attr->GetNormals()->GetName() : nullptr;
  const char *tcoords_name =
      attr->GetTCoords() ? attr->GetTCoords()->GetName() : nullptr;
  const char *tensors_name =
      attr->GetTensors() ? attr->GetTensors()->GetName() : nullptr;

  // Reindex all arrays
  int n_arrays = attr->GetNumberOfArrays();
  for (int i = 0; i < n_arrays; ++i) {
    auto *arr = attr->GetArray(i);
    out->AddArray(make_vtk_array_reindexed(arr, im));
  }

  // Restore active attributes
  if (scalars_name)
    out->SetActiveScalars(scalars_name);
  if (vectors_name)
    out->SetActiveVectors(vectors_name);
  if (normals_name)
    out->SetActiveNormals(normals_name);
  if (tcoords_name)
    out->SetActiveTCoords(tcoords_name);
  if (tensors_name)
    out->SetActiveTensors(tensors_name);

  return out;
}

auto make_vtk_point_data_reindexed(vtkPointData *attr,
                                   const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkPointData> {
  auto base = make_vtk_data_set_attributes_reindexed(attr, im);
  auto out = vtkSmartPointer<vtkPointData>::New();
  out->ShallowCopy(base);
  return out;
}

auto make_vtk_cell_data_reindexed(vtkCellData *attr,
                                  const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkCellData> {
  auto base = make_vtk_data_set_attributes_reindexed(attr, im);
  auto out = vtkSmartPointer<vtkCellData>::New();
  out->ShallowCopy(base);
  return out;
}

} // namespace tf::vtk
