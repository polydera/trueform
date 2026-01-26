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
#include <trueform/vtk/filters/non_simple_edges.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/make_boundary_edges.hpp>
#include <trueform/vtk/functions/make_non_manifold_edges.hpp>
#include <trueform/vtk/functions/make_non_simple_edges.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(non_simple_edges);

non_simple_edges::non_simple_edges() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto non_simple_edges::set_boundary_edges(bool value) -> void {
  if (_boundary_edges != value) {
    _boundary_edges = value;
    Modified();
  }
}

auto non_simple_edges::boundary_edges() const -> bool {
  return _boundary_edges;
}

auto non_simple_edges::set_non_manifold_edges(bool value) -> void {
  if (_non_manifold_edges != value) {
    _non_manifold_edges = value;
    Modified();
  }
}

auto non_simple_edges::non_manifold_edges() const -> bool {
  return _non_manifold_edges;
}

auto non_simple_edges::FillInputPortInformation(int port, vtkInformation *info)
    -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto non_simple_edges::RequestData(vtkInformation *, vtkInformationVector **input,
                                   vtkInformationVector *output) -> int {
  auto *in = polydata::SafeDownCast(vtkPolyData::GetData(input[0]));

  if (!in) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  auto *out = polydata::GetData(output);

  if (_boundary_edges && _non_manifold_edges) {
    out->ShallowCopy(make_non_simple_edges(in));
  } else if (_boundary_edges) {
    out->ShallowCopy(make_boundary_edges(in));
  } else if (_non_manifold_edges) {
    out->ShallowCopy(make_non_manifold_edges(in));
  } else {
    out->Initialize();
  }

  return 1;
}

} // namespace tf::vtk
