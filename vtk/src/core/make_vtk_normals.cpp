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
#include <trueform/vtk/core/make_vtk_normals.hpp>
#include <trueform/vtk/core/make_normals.hpp>
#include <vtkFloatArray.h>

namespace tf::vtk {

auto make_vtk_normals(const tf::unit_vectors_buffer<float, 3> &normals)
    -> vtkSmartPointer<vtkFloatArray> {
  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(3);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(normals.size()));

  auto *ptr = static_cast<float *>(arr->GetVoidPointer(0));
  auto dst = tf::make_unit_vectors<3>(tf::make_range(ptr, 3 * normals.size()));
  tf::parallel_copy(normals.unit_vectors(), dst);

  return arr;
}

auto make_vtk_normals(tf::unit_vectors_buffer<float, 3> &&normals)
    -> vtkSmartPointer<vtkFloatArray> {
  auto n = normals.size();
  auto *ptr = normals.data_buffer().release();

  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(3);
  arr->SetArray(ptr, static_cast<vtkIdType>(3 * n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);

  return arr;
}

} // namespace tf::vtk
