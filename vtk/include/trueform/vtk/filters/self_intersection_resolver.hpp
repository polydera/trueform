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

/// @brief Resolves self-intersections in a mesh by embedding intersection curves.
///
/// Takes a vtkPolyData input (must be tf::vtk::polydata from adapter filter)
/// and outputs a mesh where self-intersection curves become edges.
///
/// Output port 0: Resolved mesh
/// Output port 1: Self-intersection curves (optional, enable with set_return_curves)
class self_intersection_resolver : public vtkPolyDataAlgorithm {
public:
  static auto New() -> self_intersection_resolver *;
  vtkTypeMacro(self_intersection_resolver, vtkPolyDataAlgorithm);

  /// @brief Enable/disable intersection curves output on port 1.
  auto set_return_curves(bool enable) -> void;
  auto return_curves() const -> bool;

protected:
  self_intersection_resolver();
  ~self_intersection_resolver() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto FillOutputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  bool _return_curves = false;

  self_intersection_resolver(const self_intersection_resolver &) = delete;
  auto operator=(const self_intersection_resolver &) -> void = delete;
};

} // namespace tf::vtk
