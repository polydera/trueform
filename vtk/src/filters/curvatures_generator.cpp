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
* Author: Ziga Sajovic
*/
#include <trueform/vtk/filters/curvatures_generator.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/compute_principal_curvatures.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>

namespace tf::vtk {

vtkStandardNewMacro(curvatures_generator);

curvatures_generator::curvatures_generator() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto curvatures_generator::set_k(int value) -> void {
  if (_k != value) {
    _k = value;
    Modified();
  }
}

auto curvatures_generator::k() const -> int { return _k; }

auto curvatures_generator::set_compute_directions(bool value) -> void {
  if (_compute_directions != value) {
    _compute_directions = value;
    Modified();
  }
}

auto curvatures_generator::compute_directions() const -> bool {
  return _compute_directions;
}

auto curvatures_generator::RequestData(vtkInformation *,
                                       vtkInformationVector **input_vec,
                                       vtkInformationVector *output_vec)
    -> int {
  auto *output = polydata::GetData(output_vec, 0);
  auto *input = polydata::GetData(input_vec[0], 0);

  if (!input) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  output->ShallowCopy(input);

  tf::vtk::compute_principal_curvatures(output, _k, _compute_directions);

  return 1;
}

} // namespace tf::vtk
