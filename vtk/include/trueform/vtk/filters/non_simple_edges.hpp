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

/// @brief Extracts boundary and non-manifold edges from a mesh.
///
/// Similar to vtkFeatureEdges but focused on topological edge classification.
/// When both edge types are enabled, adds "EdgeType" cell data (0=boundary, 1=non-manifold).
///
/// Edge types:
/// - Boundary edges: edges used by only one polygon
/// - Non-manifold edges: edges used by three or more polygons
///
/// @code
/// vtkNew<tf::vtk::non_simple_edges> edges;
/// edges->SetInputConnection(adapter->GetOutputPort());
/// edges->set_boundary_edges(true);
/// edges->set_non_manifold_edges(true);
/// edges->Update();
/// @endcode
class non_simple_edges : public vtkPolyDataAlgorithm {
public:
  static auto New() -> non_simple_edges *;
  vtkTypeMacro(non_simple_edges, vtkPolyDataAlgorithm);

  /// @brief Enable/disable extraction of boundary edges.
  auto set_boundary_edges(bool value) -> void;
  auto boundary_edges() const -> bool;

  /// @brief Enable/disable extraction of non-manifold edges.
  auto set_non_manifold_edges(bool value) -> void;
  auto non_manifold_edges() const -> bool;

protected:
  non_simple_edges();
  ~non_simple_edges() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  bool _boundary_edges = true;
  bool _non_manifold_edges = true;

  non_simple_edges(const non_simple_edges &) = delete;
  auto operator=(const non_simple_edges &) -> void = delete;
};

} // namespace tf::vtk
