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
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Cleans lines by removing duplicate points, degenerate edges,
/// and reconnecting edges into continuous paths.
///
/// This filter:
/// 1. Extracts all edges from lines
/// 2. Cleans segments (merge duplicate points, remove degenerate edges)
/// 3. Reconnects edges into continuous paths
///
/// Note: Cell data cannot be preserved because edges are reconnected into paths.
///
/// @code
/// vtkNew<tf::vtk::line_cleaner> cleaner;
/// cleaner->SetInputConnection(reader->GetOutputPort());
/// cleaner->set_tolerance(1e-6f);
/// cleaner->Update();
/// @endcode
class line_cleaner : public vtkPolyDataAlgorithm {
public:
  static auto New() -> line_cleaner *;
  vtkTypeMacro(line_cleaner, vtkPolyDataAlgorithm);

  /// @brief Set distance tolerance for merging points.
  /// Default: 0 (exact duplicates only)
  auto set_tolerance(float value) -> void;
  auto tolerance() const -> float;

  /// @brief Enable/disable preserving point data arrays.
  /// Default: true
  auto set_preserve_data(bool value) -> void;
  auto preserve_data() const -> bool;

protected:
  line_cleaner();
  ~line_cleaner() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  float _tolerance = 0.f;
  bool _preserve_data = true;

  line_cleaner(const line_cleaner &) = delete;
  auto operator=(const line_cleaner &) -> void = delete;
};

} // namespace tf::vtk
