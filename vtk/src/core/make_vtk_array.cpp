/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_vtk_array.hpp>
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

} // namespace tf::vtk
