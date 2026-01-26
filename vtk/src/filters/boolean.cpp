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
#include <trueform/vtk/filters/boolean.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_boolean.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(boolean);

boolean::boolean() {
  SetNumberOfInputPorts(2);
  SetNumberOfOutputPorts(2);
}

auto boolean::set_operation(tf::boolean_op op) -> void {
  if (_operation != op) {
    _operation = op;
    Modified();
  }
}

auto boolean::operation() const -> tf::boolean_op { return _operation; }

auto boolean::set_return_curves(bool enable) -> void {
  if (_return_curves != enable) {
    _return_curves = enable;
    Modified();
  }
}

auto boolean::return_curves() const -> bool { return _return_curves; }

auto boolean::set_matrix0(vtkMatrix4x4 *m) -> void {
  if (_matrix0 != m) {
    _matrix0 = m;
    _matrix0_mtime = 0;
    Modified();
  }
}

auto boolean::matrix0() const -> vtkMatrix4x4 * { return _matrix0; }

auto boolean::set_matrix1(vtkMatrix4x4 *m) -> void {
  if (_matrix1 != m) {
    _matrix1 = m;
    _matrix1_mtime = 0;
    Modified();
  }
}

auto boolean::matrix1() const -> vtkMatrix4x4 * { return _matrix1; }

auto boolean::GetMTime() -> vtkMTimeType {
  auto mtime = Superclass::GetMTime();
  if (_matrix0)
    mtime = std::max(mtime, _matrix0->GetMTime());
  if (_matrix1)
    mtime = std::max(mtime, _matrix1->GetMTime());
  return mtime;
}

auto boolean::FillInputPortInformation(int port, vtkInformation *info) -> int {
  if (port == 0 || port == 1) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto boolean::FillOutputPortInformation(int port, vtkInformation *info) -> int {
  if (port == 0 || port == 1) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto boolean::RequestData(vtkInformation *, vtkInformationVector **input,
                          vtkInformationVector *output) -> int {
  auto *in0 = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));
  auto *in1 = polydata::SafeDownCast(vtkPolyData::GetData(input[1]));

  if (!in0 || !in1) {
    vtkErrorMacro("Both inputs must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto *out_mesh = polydata::GetData(output, 0);
  auto *out_curves = polydata::GetData(output, 1);

  if (_return_curves) {
    vtkSmartPointer<polydata> mesh;
    vtkSmartPointer<polydata> curves;

    if (_matrix0 && _matrix1) {
      std::tie(mesh, curves) = make_boolean(
          {in0, _matrix0}, {in1, _matrix1}, _operation, tf::return_curves);
    } else if (_matrix0) {
      std::tie(mesh, curves) =
          make_boolean({in0, _matrix0}, in1, _operation, tf::return_curves);
    } else if (_matrix1) {
      std::tie(mesh, curves) =
          make_boolean(in0, {in1, _matrix1}, _operation, tf::return_curves);
    } else {
      std::tie(mesh, curves) =
          make_boolean(in0, in1, _operation, tf::return_curves);
    }

    if (!mesh) {
      vtkErrorMacro("Failed to compute boolean");
      return 0;
    }

    out_mesh->ShallowCopy(mesh);
    out_curves->ShallowCopy(curves);
  } else {
    vtkSmartPointer<polydata> mesh;

    if (_matrix0 && _matrix1) {
      mesh = make_boolean({in0, _matrix0}, {in1, _matrix1}, _operation);
    } else if (_matrix0) {
      mesh = make_boolean({in0, _matrix0}, in1, _operation);
    } else if (_matrix1) {
      mesh = make_boolean(in0, {in1, _matrix1}, _operation);
    } else {
      mesh = make_boolean(in0, in1, _operation);
    }

    if (!mesh) {
      vtkErrorMacro("Failed to compute boolean");
      return 0;
    }

    out_mesh->ShallowCopy(mesh);
  }

  return 1;
}

} // namespace tf::vtk
