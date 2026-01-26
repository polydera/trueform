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
#include <vtkMatrix4x4.h>
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Computes intersection curves between two meshes.
///
/// Takes two vtkPolyData inputs (must be tf::vtk::polydata from adapter filter)
/// and optional matrices for each. Outputs the intersection curves.
class intersection_curves : public vtkPolyDataAlgorithm {
public:
  static auto New() -> intersection_curves *;
  vtkTypeMacro(intersection_curves, vtkPolyDataAlgorithm);

  /// @brief Set matrix for first input.
  auto set_matrix0(vtkMatrix4x4 *m) -> void;
  auto matrix0() const -> vtkMatrix4x4 *;

  /// @brief Set matrix for second input.
  auto set_matrix1(vtkMatrix4x4 *m) -> void;
  auto matrix1() const -> vtkMatrix4x4 *;

  auto GetMTime() -> vtkMTimeType override;

protected:
  intersection_curves();
  ~intersection_curves() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  vtkMatrix4x4 *_matrix0 = nullptr;
  vtkMatrix4x4 *_matrix1 = nullptr;
  vtkMTimeType _matrix0_mtime = 0;
  vtkMTimeType _matrix1_mtime = 0;

  intersection_curves(const intersection_curves &) = delete;
  auto operator=(const intersection_curves &) -> void = delete;
};

} // namespace tf::vtk
