/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/polydata.hpp>
#include <vtkPolyDataAlgorithm.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief VTK filter that wraps vtkPolyData into tf::vtk::polydata.
///
/// Takes any vtkPolyData input and outputs a tf::vtk::polydata with
/// cached acceleration structures.
///
/// The polydata member persists between pipeline executions, preserving
/// cached structures. Structures are only rebuilt when input data changes.
class adapter : public vtkPolyDataAlgorithm {
public:
  static auto New() -> adapter *;
  vtkTypeMacro(adapter, vtkPolyDataAlgorithm);

  /// @brief Mark output as pure triangles for optimized access.
  auto set_as_triangles(bool v) -> void;
  auto as_triangles() const -> bool;

  /// @brief Get the cached polydata with acceleration structures.
  auto cached_polydata() -> polydata * { return _polydata.Get(); }

protected:
  adapter();
  ~adapter() override = default;

  auto FillInputPortInformation(int port, vtkInformation *info) -> int override;
  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  vtkSmartPointer<polydata> _polydata;
  vtkPolyData *_input_ptr = nullptr;
  bool _as_triangles = false;

  adapter(const adapter &) = delete;
  auto operator=(const adapter &) -> void = delete;
};

} // namespace tf::vtk
