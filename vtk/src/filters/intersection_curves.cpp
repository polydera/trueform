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
#include <trueform/vtk/filters/intersection_curves.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_intersection_curves.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(intersection_curves);

intersection_curves::intersection_curves() {
  SetNumberOfInputPorts(2);
  SetNumberOfOutputPorts(1);
}

auto intersection_curves::set_matrix0(vtkMatrix4x4 *m) -> void {
  if (_matrix0 != m) {
    _matrix0 = m;
    _matrix0_mtime = 0;
    Modified();
  }
}

auto intersection_curves::matrix0() const -> vtkMatrix4x4 * {
  return _matrix0;
}

auto intersection_curves::set_matrix1(vtkMatrix4x4 *m) -> void {
  if (_matrix1 != m) {
    _matrix1 = m;
    _matrix1_mtime = 0;
    Modified();
  }
}

auto intersection_curves::matrix1() const -> vtkMatrix4x4 * {
  return _matrix1;
}

auto intersection_curves::GetMTime() -> vtkMTimeType {
  auto mtime = Superclass::GetMTime();
  if (_matrix0)
    mtime = std::max(mtime, _matrix0->GetMTime());
  if (_matrix1)
    mtime = std::max(mtime, _matrix1->GetMTime());
  return mtime;
}

auto intersection_curves::FillInputPortInformation(int port,
                                                   vtkInformation *info)
    -> int {
  if (port == 0 || port == 1) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto intersection_curves::RequestData(vtkInformation *,
                                      vtkInformationVector **input,
                                      vtkInformationVector *output) -> int {
  auto *in0 = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));
  auto *in1 = polydata::SafeDownCast(vtkPolyData::GetData(input[1]));

  if (!in0 || !in1) {
    vtkErrorMacro("Both inputs must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  vtkSmartPointer<polydata> result;

  if (_matrix0 && _matrix1) {
    result = make_intersection_curves({in0, _matrix0}, {in1, _matrix1});
  } else if (_matrix0) {
    result = make_intersection_curves({in0, _matrix0}, in1);
  } else if (_matrix1) {
    result = make_intersection_curves(in0, {in1, _matrix1});
  } else {
    result = make_intersection_curves(in0, in1);
  }

  if (!result) {
    vtkErrorMacro("Failed to compute intersection curves");
    return 0;
  }

  auto *out = polydata::GetData(output);
  out->ShallowCopy(result);

  return 1;
}

} // namespace tf::vtk
