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
#include <trueform/vtk/filters/isocontours.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_isocontours.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(isocontours);

isocontours::isocontours() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto isocontours::set_scalars_name(const std::string &name) -> void {
  if (_scalars_name != name) {
    _scalars_name = name;
    Modified();
  }
}

auto isocontours::scalars_name() const -> const std::string & {
  return _scalars_name;
}

auto isocontours::set_cut_values(const std::vector<float> &values) -> void {
  _cut_values = values;
  Modified();
}

auto isocontours::set_cut_values(std::vector<float> &&values) -> void {
  _cut_values = std::move(values);
  Modified();
}

auto isocontours::cut_values() const -> const std::vector<float> & {
  return _cut_values;
}

auto isocontours::add_cut_value(float value) -> void {
  _cut_values.push_back(value);
  Modified();
}

auto isocontours::clear_cut_values() -> void {
  _cut_values.clear();
  Modified();
}

auto isocontours::FillInputPortInformation(int port, vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto isocontours::RequestData(vtkInformation *, vtkInformationVector **input,
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

  const char *name = _scalars_name.empty() ? nullptr : _scalars_name.c_str();
  auto result = make_isocontours(in, name, _cut_values);

  if (!result) {
    vtkErrorMacro("Failed to compute isocontours");
    return 0;
  }

  auto *out = polydata::GetData(output);
  out->ShallowCopy(result);

  return 1;
}

} // namespace tf::vtk
