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

/// @brief Computes normals and optionally orients faces.
///
/// This filter computes cell normals and optionally point normals.
/// It can also orient faces for consistent winding before computing normals.
///
/// Operations are performed in order:
/// 1. Orient faces (if enabled): positive orientation or just consistent
/// 2. Compute cell normals (always)
/// 3. Compute point normals (if enabled)
///
/// By default, orient_faces and compute_point_normals are enabled,
/// positive_orientation is disabled.
///
/// @code
/// vtkNew<tf::vtk::normals_generator> filter;
/// filter->SetInputConnection(reader->GetOutputPort());
/// filter->set_orient_faces(true);
/// filter->set_positive_orientation(true);  // outward-facing normals
/// filter->set_compute_point_normals(true);
/// filter->Update();
/// @endcode
class normals_generator : public vtkPolyDataAlgorithm {
public:
  static auto New() -> normals_generator *;
  vtkTypeMacro(normals_generator, vtkPolyDataAlgorithm);

  /// @brief Enable/disable consistent face orientation.
  /// Default: true
  auto set_orient_faces(bool value) -> void;
  auto orient_faces() const -> bool;

  /// @brief Enable/disable positive orientation (outward-facing normals).
  /// When enabled, ensures signed volume is positive after consistent
  /// orientation. Implies orient_faces. Default: false
  auto set_positive_orientation(bool value) -> void;
  auto positive_orientation() const -> bool;

  /// @brief Enable/disable point normal computation.
  /// Default: true
  auto set_compute_point_normals(bool value) -> void;
  auto compute_point_normals() const -> bool;

protected:
  normals_generator();
  ~normals_generator() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  bool _orient_faces = true;
  bool _positive_orientation = false;
  bool _compute_point_normals = true;

  normals_generator(const normals_generator &) = delete;
  auto operator=(const normals_generator &) -> void = delete;
};

} // namespace tf::vtk
