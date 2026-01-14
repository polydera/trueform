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

/// @brief Extracts boundary paths (loops/chains) from a mesh.
///
/// Returns polylines representing the boundary of the mesh, using
/// the original vertex IDs from the input mesh.
///
/// @code
/// vtkNew<tf::vtk::boundary_paths> boundary;
/// boundary->SetInputConnection(reader->GetOutputPort());
/// boundary->Update();
/// // Output: polylines using input mesh points
/// @endcode
class boundary_paths : public vtkPolyDataAlgorithm {
public:
  static auto New() -> boundary_paths *;
  vtkTypeMacro(boundary_paths, vtkPolyDataAlgorithm);

protected:
  boundary_paths();
  ~boundary_paths() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  boundary_paths(const boundary_paths &) = delete;
  auto operator=(const boundary_paths &) -> void = delete;
};

} // namespace tf::vtk
