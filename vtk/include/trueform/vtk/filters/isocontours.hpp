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
#pragma once
#include <string>
#include <vector>
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Extracts isocontour curves from scalar fields on a mesh.
///
/// Takes a vtkPolyData input with point scalars and outputs polylines
/// at the specified cut values.
///
/// @code
/// vtkNew<tf::vtk::isocontours> iso;
/// iso->SetInputConnection(reader->GetOutputPort());
/// iso->set_scalars_name("temperature");
/// iso->set_cut_values({20.0f, 25.0f, 30.0f});
/// iso->Update();
/// @endcode
class isocontours : public vtkPolyDataAlgorithm {
public:
  static auto New() -> isocontours *;
  vtkTypeMacro(isocontours, vtkPolyDataAlgorithm);

  /// @brief Set the name of the point scalars array to use.
  auto set_scalars_name(const std::string &name) -> void;
  auto scalars_name() const -> const std::string &;

  /// @brief Set the cut values for isocontour extraction.
  auto set_cut_values(const std::vector<float> &values) -> void;
  auto set_cut_values(std::vector<float> &&values) -> void;
  auto cut_values() const -> const std::vector<float> &;

  /// @brief Add a single cut value.
  auto add_cut_value(float value) -> void;

  /// @brief Clear all cut values.
  auto clear_cut_values() -> void;

protected:
  isocontours();
  ~isocontours() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  std::string _scalars_name;
  std::vector<float> _cut_values;

  isocontours(const isocontours &) = delete;
  auto operator=(const isocontours &) -> void = delete;
};

} // namespace tf::vtk
