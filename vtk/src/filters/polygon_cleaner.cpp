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
#include <trueform/vtk/filters/polygon_cleaner.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/cleaned_polygons.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>

namespace tf::vtk {

vtkStandardNewMacro(polygon_cleaner);

polygon_cleaner::polygon_cleaner() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto polygon_cleaner::set_tolerance(float value) -> void {
  if (_tolerance != value) {
    _tolerance = value;
    Modified();
  }
}

auto polygon_cleaner::tolerance() const -> float { return _tolerance; }

auto polygon_cleaner::set_preserve_data(bool value) -> void {
  if (_preserve_data != value) {
    _preserve_data = value;
    Modified();
  }
}

auto polygon_cleaner::preserve_data() const -> bool { return _preserve_data; }

auto polygon_cleaner::RequestData(vtkInformation *,
                                  vtkInformationVector **input_vec,
                                  vtkInformationVector *output_vec) -> int {
  auto *input = polydata::SafeDownCast(vtkPolyData::GetData(input_vec[0], 0));
  auto *output = polydata::GetData(output_vec, 0);

  if (!input) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto cleaned = cleaned_polygons(input, _tolerance, _preserve_data);
  if (cleaned) {
    output->ShallowCopy(cleaned);
  }

  return 1;
}

} // namespace tf::vtk
