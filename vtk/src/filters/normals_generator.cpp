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
#include <trueform/vtk/filters/normals_generator.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/compute_cell_normals.hpp>
#include <trueform/vtk/functions/compute_point_normals.hpp>
#include <trueform/vtk/functions/ensure_positive_orientation.hpp>
#include <trueform/vtk/functions/orient_faces_consistently.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>

namespace tf::vtk {

vtkStandardNewMacro(normals_generator);

normals_generator::normals_generator() {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto normals_generator::set_orient_faces(bool value) -> void {
  if (_orient_faces != value) {
    _orient_faces = value;
    Modified();
  }
}

auto normals_generator::orient_faces() const -> bool { return _orient_faces; }

auto normals_generator::set_positive_orientation(bool value) -> void {
  if (_positive_orientation != value) {
    _positive_orientation = value;
    Modified();
  }
}

auto normals_generator::positive_orientation() const -> bool {
  return _positive_orientation;
}

auto normals_generator::set_compute_point_normals(bool value) -> void {
  if (_compute_point_normals != value) {
    _compute_point_normals = value;
    Modified();
  }
}

auto normals_generator::compute_point_normals() const -> bool {
  return _compute_point_normals;
}

auto normals_generator::RequestData(vtkInformation *,
                                    vtkInformationVector **input_vec,
                                    vtkInformationVector *output_vec) -> int {
  auto *output = polydata::GetData(output_vec, 0);
  auto *input = vtkPolyData::GetData(input_vec[0], 0);

  if (!input) {
    vtkErrorMacro("Input must be tf::vtk::polydata (use adapter filter)");
    return 0;
  }

  output->ShallowCopy(input);

  if (_positive_orientation) {
    ensure_positive_orientation(output);
  } else if (_orient_faces) {
    orient_faces_consistently(output);
  }

  tf::vtk::compute_cell_normals(output);

  if (_compute_point_normals) {
    tf::vtk::compute_point_normals(output);
  }

  return 1;
}

} // namespace tf::vtk
