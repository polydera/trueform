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
#include <trueform/vtk/filters/isobands.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_isobands.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(isobands);

isobands::isobands() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(2);
}

auto isobands::set_scalars_name(const std::string &name) -> void {
  if (_scalars_name != name) {
    _scalars_name = name;
    Modified();
  }
}

auto isobands::scalars_name() const -> const std::string & {
  return _scalars_name;
}

auto isobands::set_cut_values(const std::vector<float> &values) -> void {
  _cut_values = values;
  Modified();
}

auto isobands::set_cut_values(std::vector<float> &&values) -> void {
  _cut_values = std::move(values);
  Modified();
}

auto isobands::cut_values() const -> const std::vector<float> & {
  return _cut_values;
}

auto isobands::set_selected_bands(const std::vector<int> &bands) -> void {
  _selected_bands = bands;
  Modified();
}

auto isobands::set_selected_bands(std::vector<int> &&bands) -> void {
  _selected_bands = std::move(bands);
  Modified();
}

auto isobands::selected_bands() const -> const std::vector<int> & {
  return _selected_bands;
}

auto isobands::set_return_curves(bool value) -> void {
  if (_return_curves != value) {
    _return_curves = value;
    Modified();
  }
}

auto isobands::return_curves() const -> bool { return _return_curves; }

auto isobands::FillInputPortInformation(int port, vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto isobands::FillOutputPortInformation(int port, vtkInformation *info)
    -> int {
  if (port == 0 || port == 1) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto isobands::RequestData(vtkInformation *, vtkInformationVector **input,
                           vtkInformationVector *output) -> int {
  auto *in = vtkPolyData::GetData(input[0]);

  if (!in) {
    vtkErrorMacro("Input is null");
    return 0;
  }

  if (_cut_values.empty()) {
    vtkErrorMacro("No cut values specified");
    return 0;
  }

  if (_selected_bands.empty()) {
    vtkErrorMacro("No bands selected");
    return 0;
  }

  const char *name = _scalars_name.empty() ? nullptr : _scalars_name.c_str();

  auto *out0 = polydata::GetData(output, 0);
  auto *out1 = polydata::GetData(output, 1);

  if (_return_curves) {
    auto [bands, curves] =
        make_isobands(in, name, _cut_values, _selected_bands, tf::return_curves);

    if (!bands) {
      vtkErrorMacro("Failed to compute isobands");
      return 0;
    }

    out0->ShallowCopy(bands);
    out1->ShallowCopy(curves);
  } else {
    auto bands = make_isobands(in, name, _cut_values, _selected_bands);

    if (!bands) {
      vtkErrorMacro("Failed to compute isobands");
      return 0;
    }

    out0->ShallowCopy(bands);
  }

  return 1;
}

} // namespace tf::vtk
