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

/// @brief Triangulates all polygons using ear-cutting.
///
/// This filter converts arbitrary polygons (quads, n-gons) to triangles.
/// Point data is optionally preserved; cell data is not preserved since
/// face count changes.
///
/// @code
/// vtkNew<tf::vtk::triangulator> filter;
/// filter->SetInputConnection(reader->GetOutputPort());
/// filter->set_preserve_point_data(true);
/// filter->Update();
/// @endcode
class triangulator : public vtkPolyDataAlgorithm {
public:
  static auto New() -> triangulator *;
  vtkTypeMacro(triangulator, vtkPolyDataAlgorithm);

  /// @brief Enable/disable preserving point data arrays.
  /// Default: true
  auto set_preserve_point_data(bool value) -> void;
  auto preserve_point_data() const -> bool;

protected:
  triangulator();
  ~triangulator() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  bool _preserve_point_data = true;

  triangulator(const triangulator &) = delete;
  auto operator=(const triangulator &) -> void = delete;
};

} // namespace tf::vtk
