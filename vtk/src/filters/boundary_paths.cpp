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
#include <trueform/vtk/filters/boundary_paths.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_boundary_paths.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(boundary_paths);

boundary_paths::boundary_paths() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto boundary_paths::FillInputPortInformation(int port, vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto boundary_paths::RequestData(vtkInformation *, vtkInformationVector **input,
                                 vtkInformationVector *output) -> int {
  auto *in = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));

  if (!in) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto *out = polydata::GetData(output);
  out->ShallowCopy(make_boundary_paths(in));

  return 1;
}

} // namespace tf::vtk
