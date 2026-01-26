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
#include <trueform/vtk/filters/connected_components.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_connected_components.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(connected_components);

connected_components::connected_components() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto connected_components::set_connectivity(tf::connectivity_type type)
    -> void {
  if (_connectivity != type) {
    _connectivity = type;
    Modified();
  }
}

auto connected_components::connectivity() const -> tf::connectivity_type {
  return _connectivity;
}

auto connected_components::n_components() const -> int { return _n_components; }

auto connected_components::FillInputPortInformation(int port,
                                                    vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto connected_components::RequestData(vtkInformation *,
                                       vtkInformationVector **input,
                                       vtkInformationVector *output) -> int {
  auto *in = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));

  if (!in) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto [result, n] = make_connected_components(in, _connectivity);

  if (!result) {
    vtkErrorMacro("Failed to compute connected components");
    return 0;
  }

  _n_components = n;

  auto *out = polydata::GetData(output);
  out->ShallowCopy(result);

  return 1;
}

} // namespace tf::vtk
