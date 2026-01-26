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
#include <trueform/cut/boolean_op.hpp>
#include <vtkMatrix4x4.h>
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Computes boolean operations between two meshes.
///
/// Takes two vtkPolyData inputs (must be tf::vtk::polydata from adapter filter)
/// and optional matrices for each.
///
/// Output port 0: Result mesh
/// Output port 1: Intersection curves (optional, enable with set_return_curves)
class boolean : public vtkPolyDataAlgorithm {
public:
  static auto New() -> boolean *;
  vtkTypeMacro(boolean, vtkPolyDataAlgorithm);

  /// @brief Set boolean operation type.
  auto set_operation(tf::boolean_op op) -> void;
  auto operation() const -> tf::boolean_op;

  /// @brief Enable/disable intersection curves output on port 1.
  auto set_return_curves(bool enable) -> void;
  auto return_curves() const -> bool;

  /// @brief Set matrix for first input.
  auto set_matrix0(vtkMatrix4x4 *m) -> void;
  auto matrix0() const -> vtkMatrix4x4 *;

  /// @brief Set matrix for second input.
  auto set_matrix1(vtkMatrix4x4 *m) -> void;
  auto matrix1() const -> vtkMatrix4x4 *;

  auto GetMTime() -> vtkMTimeType override;

protected:
  boolean();
  ~boolean() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto FillOutputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  tf::boolean_op _operation = tf::boolean_op::merge;
  bool _return_curves = false;

  vtkMatrix4x4 *_matrix0 = nullptr;
  vtkMatrix4x4 *_matrix1 = nullptr;
  vtkMTimeType _matrix0_mtime = 0;
  vtkMTimeType _matrix1_mtime = 0;

  boolean(const boolean &) = delete;
  auto operator=(const boolean &) -> void = delete;
};

} // namespace tf::vtk
