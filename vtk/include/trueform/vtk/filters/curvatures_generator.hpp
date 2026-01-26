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
* Author: Ziga Sajovic
*/
#pragma once
#include <vtkPolyDataAlgorithm.h>

namespace tf::vtk {

/// @brief Computes principal curvatures at mesh vertices.
///
/// This filter computes principal curvature values (K1, K2) at each vertex
/// using quadric fitting on k-ring neighborhoods. Optionally computes
/// principal directions (D1, D2) as well.
///
/// @code
/// vtkNew<tf::vtk::curvatures_generator> filter;
/// filter->SetInputConnection(adapter->GetOutputPort());
/// filter->set_k(2);                     // neighborhood rings
/// filter->set_compute_directions(true); // also compute directions
/// filter->Update();
/// @endcode
class curvatures_generator : public vtkPolyDataAlgorithm {
public:
  static auto New() -> curvatures_generator *;
  vtkTypeMacro(curvatures_generator, vtkPolyDataAlgorithm);

  /// @brief Set k-ring neighborhood size.
  /// Default: 2
  auto set_k(int value) -> void;
  auto k() const -> int;

  /// @brief Enable/disable principal direction computation.
  /// When enabled, adds D1 and D2 vector arrays to point data.
  /// Default: false
  auto set_compute_directions(bool value) -> void;
  auto compute_directions() const -> bool;

protected:
  curvatures_generator();
  ~curvatures_generator() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  int _k = 2;
  bool _compute_directions = false;

  curvatures_generator(const curvatures_generator &) = delete;
  auto operator=(const curvatures_generator &) -> void = delete;
};

} // namespace tf::vtk
