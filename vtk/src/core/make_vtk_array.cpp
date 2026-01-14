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
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkSignedCharArray.h>

namespace tf::vtk {

auto make_vtk_array(const tf::buffer<std::int8_t> &buffer)
    -> vtkSmartPointer<vtkSignedCharArray> {
  auto arr = vtkSmartPointer<vtkSignedCharArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(buffer.size()));
  tf::parallel_copy(buffer, tf::make_range(arr->GetPointer(0), buffer.size()));
  return arr;
}

auto make_vtk_array(tf::buffer<std::int8_t> &&buffer)
    -> vtkSmartPointer<vtkSignedCharArray> {
  auto n = buffer.size();
  auto *ptr = buffer.release();

  auto arr = vtkSmartPointer<vtkSignedCharArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetArray(ptr, static_cast<vtkIdType>(n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  return arr;
}

auto make_vtk_array(const tf::buffer<int> &buffer)
    -> vtkSmartPointer<vtkIntArray> {
  auto arr = vtkSmartPointer<vtkIntArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(buffer.size()));
  tf::parallel_copy(buffer, tf::make_range(arr->GetPointer(0), buffer.size()));
  return arr;
}

auto make_vtk_array(tf::buffer<int> &&buffer)
    -> vtkSmartPointer<vtkIntArray> {
  auto n = buffer.size();
  auto *ptr = buffer.release();

  auto arr = vtkSmartPointer<vtkIntArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetArray(ptr, static_cast<vtkIdType>(n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  return arr;
}

auto make_vtk_array(const tf::buffer<vtkIdType> &buffer)
    -> vtkSmartPointer<vtkIdTypeArray> {
  auto arr = vtkSmartPointer<vtkIdTypeArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(buffer.size()));
  tf::parallel_copy(buffer, tf::make_range(arr->GetPointer(0), buffer.size()));
  return arr;
}

auto make_vtk_array(tf::buffer<vtkIdType> &&buffer)
    -> vtkSmartPointer<vtkIdTypeArray> {
  auto n = buffer.size();
  auto *ptr = buffer.release();

  auto arr = vtkSmartPointer<vtkIdTypeArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetArray(ptr, static_cast<vtkIdType>(n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  return arr;
}

auto make_vtk_array(const tf::buffer<float> &buffer)
    -> vtkSmartPointer<vtkFloatArray> {
  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(buffer.size()));
  tf::parallel_copy(buffer, tf::make_range(arr->GetPointer(0), buffer.size()));
  return arr;
}

auto make_vtk_array(tf::buffer<float> &&buffer)
    -> vtkSmartPointer<vtkFloatArray> {
  auto n = buffer.size();
  auto *ptr = buffer.release();

  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(1);
  arr->SetArray(ptr, static_cast<vtkIdType>(n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  return arr;
}

auto make_vtk_array(const tf::unit_vectors_buffer<float, 3> &buffer)
    -> vtkSmartPointer<vtkFloatArray> {
  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(3);
  arr->SetNumberOfTuples(static_cast<vtkIdType>(buffer.size()));

  auto *ptr = static_cast<float *>(arr->GetVoidPointer(0));
  auto dst = tf::make_unit_vectors<3>(tf::make_range(ptr, 3 * buffer.size()));
  tf::parallel_copy(buffer.unit_vectors(), dst);

  return arr;
}

auto make_vtk_array(tf::unit_vectors_buffer<float, 3> &&buffer)
    -> vtkSmartPointer<vtkFloatArray> {
  auto n = buffer.size();
  auto *ptr = buffer.data_buffer().release();

  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetNumberOfComponents(3);
  arr->SetArray(ptr, static_cast<vtkIdType>(3 * n), 0,
                vtkAbstractArray::VTK_DATA_ARRAY_DELETE);

  return arr;
}

} // namespace tf::vtk
