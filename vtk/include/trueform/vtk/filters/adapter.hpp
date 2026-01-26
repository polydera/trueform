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

  adapter(const adapter &) = delete;
  auto operator=(const adapter &) -> void = delete;
};

} // namespace tf::vtk
