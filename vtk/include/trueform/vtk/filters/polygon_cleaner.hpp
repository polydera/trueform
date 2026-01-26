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

/// @brief Cleans polygons by removing duplicate points and degenerate faces.
///
/// This filter removes duplicate vertices (within a tolerance) and any
/// degenerate faces that result from the vertex merging. Point and cell
/// data arrays can optionally be remapped.
///
/// @code
/// vtkNew<tf::vtk::polygon_cleaner> cleaner;
/// cleaner->SetInputConnection(reader->GetOutputPort());
/// cleaner->set_tolerance(1e-6f);
/// cleaner->Update();
/// @endcode
class polygon_cleaner : public vtkPolyDataAlgorithm {
public:
  static auto New() -> polygon_cleaner *;
  vtkTypeMacro(polygon_cleaner, vtkPolyDataAlgorithm);

  /// @brief Set distance tolerance for merging points.
  /// Default: 0 (exact duplicates only)
  auto set_tolerance(float value) -> void;
  auto tolerance() const -> float;

  /// @brief Enable/disable preserving point and cell data arrays.
  /// Default: true
  auto set_preserve_data(bool value) -> void;
  auto preserve_data() const -> bool;

protected:
  polygon_cleaner();
  ~polygon_cleaner() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  float _tolerance = 0.f;
  bool _preserve_data = true;

  polygon_cleaner(const polygon_cleaner &) = delete;
  auto operator=(const polygon_cleaner &) -> void = delete;
};

} // namespace tf::vtk
