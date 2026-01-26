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

/// @brief Extracts isoband regions from scalar fields on a mesh.
///
/// Takes a vtkPolyData input with point scalars and outputs polygon regions
/// between specified cut values. Optionally outputs boundary curves on port 1.
///
/// @code
/// vtkNew<tf::vtk::isobands> bands;
/// bands->SetInputConnection(reader->GetOutputPort());
/// bands->set_scalars_name("temperature");
/// bands->set_cut_values({20.0f, 25.0f, 30.0f});
/// bands->set_selected_bands({0, 1});  // select bands between cuts
/// bands->set_return_curves(true);
/// bands->Update();
/// // Port 0: isoband polygons with labels
/// // Port 1: boundary curves (if return_curves is true)
/// @endcode
class isobands : public vtkPolyDataAlgorithm {
public:
  static auto New() -> isobands *;
  vtkTypeMacro(isobands, vtkPolyDataAlgorithm);

  /// @brief Set the name of the point scalars array to use.
  auto set_scalars_name(const std::string &name) -> void;
  auto scalars_name() const -> const std::string &;

  /// @brief Set the cut values for isoband extraction.
  auto set_cut_values(const std::vector<float> &values) -> void;
  auto set_cut_values(std::vector<float> &&values) -> void;
  auto cut_values() const -> const std::vector<float> &;

  /// @brief Set which bands to extract (indices into cut_values intervals).
  /// Band i is the region between cut_values[i] and cut_values[i+1].
  auto set_selected_bands(const std::vector<int> &bands) -> void;
  auto set_selected_bands(std::vector<int> &&bands) -> void;
  auto selected_bands() const -> const std::vector<int> &;

  /// @brief Enable output of boundary curves on port 1.
  auto set_return_curves(bool value) -> void;
  auto return_curves() const -> bool;

protected:
  isobands();
  ~isobands() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto FillOutputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  std::string _scalars_name;
  std::vector<float> _cut_values;
  std::vector<int> _selected_bands;
  bool _return_curves = false;

  isobands(const isobands &) = delete;
  auto operator=(const isobands &) -> void = delete;
};

} // namespace tf::vtk
