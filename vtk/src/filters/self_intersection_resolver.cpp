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
#include <trueform/vtk/filters/self_intersection_resolver.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/resolved_self_intersections.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(self_intersection_resolver);

self_intersection_resolver::self_intersection_resolver() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(2);
}

auto self_intersection_resolver::set_return_curves(bool enable) -> void {
  if (_return_curves != enable) {
    _return_curves = enable;
    Modified();
  }
}

auto self_intersection_resolver::return_curves() const -> bool {
  return _return_curves;
}

auto self_intersection_resolver::FillInputPortInformation(int port,
                                                          vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto self_intersection_resolver::FillOutputPortInformation(int port,
                                                           vtkInformation *info)
    -> int {
  if (port == 0 || port == 1) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto self_intersection_resolver::RequestData(vtkInformation *,
                                             vtkInformationVector **input,
                                             vtkInformationVector *output)
    -> int {
  auto *in = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));

  if (!in) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto *out_mesh = polydata::GetData(output, 0);
  auto *out_curves = polydata::GetData(output, 1);

  if (_return_curves) {
    auto [mesh, curves] = resolved_self_intersections(in, tf::return_curves);

    if (!mesh) {
      vtkErrorMacro("Failed to resolve self-intersections");
      return 0;
    }

    out_mesh->ShallowCopy(mesh);
    out_curves->ShallowCopy(curves);
  } else {
    auto mesh = resolved_self_intersections(in);

    if (!mesh) {
      vtkErrorMacro("Failed to resolve self-intersections");
      return 0;
    }

    out_mesh->ShallowCopy(mesh);
  }

  return 1;
}

} // namespace tf::vtk
