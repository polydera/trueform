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
#include <trueform/topology/connectivity_type.hpp>
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Labels connected components in a mesh.
///
/// Takes a vtkPolyData input (must be tf::vtk::polydata from adapter filter)
/// and adds "ComponentLabel" cell data with component IDs.
///
/// Connectivity types:
/// - manifold_edge: only through manifold edges (separates at boundaries/non-manifold)
/// - edge: through any shared edge
/// - vertex: through any shared vertex (most permissive)
///
/// @code
/// vtkNew<tf::vtk::connected_components> cc;
/// cc->SetInputConnection(adapter->GetOutputPort());
/// cc->set_connectivity(tf::connectivity_type::manifold_edge);
/// cc->Update();
/// int n = cc->n_components();
/// @endcode
class connected_components : public vtkPolyDataAlgorithm {
public:
  static auto New() -> connected_components *;
  vtkTypeMacro(connected_components, vtkPolyDataAlgorithm);

  /// @brief Set connectivity type for component detection.
  auto set_connectivity(tf::connectivity_type type) -> void;
  auto connectivity() const -> tf::connectivity_type;

  /// @brief Get number of components found (valid after Update).
  auto n_components() const -> int;

protected:
  connected_components();
  ~connected_components() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  tf::connectivity_type _connectivity = tf::connectivity_type::manifold_edge;
  int _n_components = 0;

  connected_components(const connected_components &) = delete;
  auto operator=(const connected_components &) -> void = delete;
};

} // namespace tf::vtk
