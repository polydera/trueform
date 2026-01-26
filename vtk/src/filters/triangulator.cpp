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
#include <trueform/vtk/filters/triangulator.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/triangulated.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>

namespace tf::vtk {

vtkStandardNewMacro(triangulator);

triangulator::triangulator() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto triangulator::set_preserve_point_data(bool value) -> void {
  if (_preserve_point_data != value) {
    _preserve_point_data = value;
    Modified();
  }
}

auto triangulator::preserve_point_data() const -> bool {
  return _preserve_point_data;
}

auto triangulator::RequestData(vtkInformation *,
                               vtkInformationVector **input_vec,
                               vtkInformationVector *output_vec) -> int {
  auto *output = polydata::GetData(output_vec, 0);
  auto *input = polydata::GetData(input_vec[0], 0);

  if (!input) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto result = triangulated(input, _preserve_point_data);
  if (!result) {
    vtkErrorMacro("Triangulation failed");
    return 0;
  }

  output->ShallowCopy(result);
  return 1;
}

} // namespace tf::vtk
