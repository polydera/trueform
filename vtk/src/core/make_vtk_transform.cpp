/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_vtk_transform.hpp>
#include <trueform/vtk/core/make_vtk_matrix.hpp>
#include <vtkTransform.h>

namespace tf::vtk {

auto make_vtk_transform(const tf::transformation<float, 3> &transform)
    -> vtkSmartPointer<vtkTransform> {
  auto vtk_transform = vtkSmartPointer<vtkTransform>::New();
  vtk_transform->SetMatrix(make_vtk_matrix(transform));
  return vtk_transform;
}

auto make_vtk_transform(const tf::transformation<double, 3> &transform)
    -> vtkSmartPointer<vtkTransform> {
  auto vtk_transform = vtkSmartPointer<vtkTransform>::New();
  vtk_transform->SetMatrix(make_vtk_matrix(transform));
  return vtk_transform;
}

auto fill_vtk_transform(vtkTransform *transform,
                        const tf::transformation<float, 3> &tf_transform) -> void {
  fill_vtk_matrix(transform->GetMatrix(), tf_transform);
  transform->Modified();
}

auto fill_vtk_transform(vtkTransform *transform,
                        const tf::transformation<double, 3> &tf_transform) -> void {
  fill_vtk_matrix(transform->GetMatrix(), tf_transform);
  transform->Modified();
}

} // namespace tf::vtk
